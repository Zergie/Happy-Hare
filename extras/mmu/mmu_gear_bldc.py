# Happy Hare MMU Software
#
# Copyright (C) 2022-2026  moggieuk#6538 (discord)
#                          moggieuk@hotmail.com
#
# Goal: Control BLDC gear motion, tachometer feedback and calibration.
#
#
# (\_/)
# ( *,*)
# (")_(") Happy Hare Ready
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#

from types import MethodType
import inspect
import math
import logging
import traceback
from .mmu_utils import MmuError
from .mmu_pin_queue import MmuPinRequestQueue as BldcPinRequestQueue
from .. import pulse_counter
from ..pwm_tool import MCU_queued_pwm

EPSILON = 1e-6
INFINITY = float('inf')

class MotionDescriptor:
    """Base motion descriptor with direction and speed mode."""
    def __init__(self, print_time, speed_mm_s, pid_enable, direction=1, duration=0.):
        self.speed_mm_s = speed_mm_s
        self.pid_enable = pid_enable
        self.print_time = print_time
        self.direction = direction  # 1=forward, -1=backward
        self.duration = duration   # Active window length in seconds
        self.pwm = None  # Only MotionPwmDirect sets this; indicates PWM-direct dispatch mode
        self.trackback = traceback.format_stack()

class MotionStop(MotionDescriptor):
    """Terminal stop descriptor (speed=0, pid=False, direction-agnostic)."""
    def __init__(self, print_time):
        super().__init__(print_time, 0., False)
        self.duration = INFINITY  # Never expires via pruning

class MotionPwmDirect(MotionDescriptor):
    """PWM-direct mode descriptor (for brake and other direct PWM control)."""
    def __init__(self, print_time, pwm, duration, direction=1):
        super().__init__(print_time, 0., False, direction=direction, duration=duration)
        self.pwm = pwm  # Set PWM directly (base class sets pwm=None)


class MotionTorqueAssist(MotionPwmDirect):
    """Fixed-PWM stall assist for synchronized motion below stable motor RPM."""


class MotionTrapzoid(MotionDescriptor):
    """Trapezoidal accel/decel profile (for process_move sync)."""
    def __init__(self, print_time, start_speed_mm_s, end_speed_mm_s, accel_mm_s2, *, direction=1):
        duration = (abs(end_speed_mm_s - start_speed_mm_s) / accel_mm_s2
                    if accel_mm_s2 > EPSILON else 0.)
        super().__init__(print_time, start_speed_mm_s, True, direction=direction, duration=duration)
        self.start_speed_mm_s = start_speed_mm_s
        self.end_speed_mm_s = end_speed_mm_s
        self.accel_mm_s2 = accel_mm_s2
        self.end_print_time = print_time + duration

    def get_speed(self, print_time):
        """Compute speed at given print_time within profile."""
        if print_time <= self.print_time:
            return self.start_speed_mm_s
        if print_time >= self.end_print_time:
            return self.end_speed_mm_s
        accel_time = print_time - self.print_time
        accel_speed = accel_time * self.accel_mm_s2
        if self.end_speed_mm_s > self.start_speed_mm_s:
            return self.start_speed_mm_s + accel_speed
        return self.start_speed_mm_s - accel_speed


class PositionFeedForwardState:
    """Learned timing and coast state for one physical motor direction."""

    def __init__(self, initial_stop_lead_counts, learning_weight):
        self.effective_frequency = 0.
        self.commanded_rpm = 0.
        self.stop_lead_counts = initial_stop_lead_counts
        self.learning_weight = learning_weight

    def estimate_frequency(self, commanded_rpm):
        if self.effective_frequency <= EPSILON or self.commanded_rpm <= EPSILON:
            return 0.
        return self.effective_frequency * commanded_rpm / self.commanded_rpm

    def learn_frequency(self, observed_frequency, commanded_rpm):
        if observed_frequency <= EPSILON or commanded_rpm <= EPSILON:
            return
        if self.effective_frequency <= EPSILON:
            self.effective_frequency = observed_frequency
        else:
            frequency_at_commanded_rpm = self.estimate_frequency(commanded_rpm)
            self.effective_frequency = (
                frequency_at_commanded_rpm
                + self.learning_weight
                * (observed_frequency - frequency_at_commanded_rpm)
            )
        self.commanded_rpm = commanded_rpm

    def learn_stop_lead(
            self, applied_lead_counts, position_error_counts, max_lead_counts):
        learned_lead_counts = (
            applied_lead_counts
            + self.learning_weight * position_error_counts
        )
        self.stop_lead_counts = min(
            max_lead_counts, max(0., learned_lead_counts)
        )


class FiniteMoveTracker:
    """Tachometer position state for one standalone finite BLDC move."""

    STANDSTILL_SAMPLE_COUNT = 2

    def __init__(self, requested_distance_mm, start_print_time, target_count_delta):
        self.requested_distance_mm = requested_distance_mm
        self.start_print_time = start_print_time
        self.target_count_delta = target_count_delta
        self.start_count = None
        self.cruise_start_time = start_print_time
        self.no_progress_deadline = INFINITY
        self.last_progress_count = 0.
        self.last_progress_time = start_print_time
        self.filtered_frequency = 0.
        self.target_rpm = 0.
        self.direction = 1
        self.learned_stop_lead_counts = 0.
        self.stop_lead_counts = 0.
        self.has_frequency_seed = False
        self.nominal_target_print_time = None
        self.target_print_time = None
        self.cruise_descriptor = None
        self.stop_descriptor = None
        self.endpoint_locked = False
        self.stop_dispatched = False
        self.completed = False
        self.aborted = False
        self.learning_completed = False
        self.actual_stop_print_time = None
        self.standstill_sample_count = 0
        self.last_standstill_sample_time = None
        self.settled_count = None
        self.settled_sample_time = None
        self.settle_timed_out = False

    def set_start_count(self, count):
        self.start_count = count
        self.last_progress_count = 0.
        self.last_progress_time = self.start_print_time

    def observe_progress(self, travelled_counts, count_time, filter_weight):
        count_delta = travelled_counts - self.last_progress_count
        time_delta = count_time - self.last_progress_time
        if count_delta <= 0. or time_delta <= 0.:
            return None

        sample_frequency = count_delta / time_delta
        if self.filtered_frequency <= EPSILON:
            self.filtered_frequency = sample_frequency
        else:
            self.filtered_frequency += filter_weight * (
                sample_frequency - self.filtered_frequency
            )
        self.last_progress_count = travelled_counts
        self.last_progress_time = count_time
        return self.filtered_frequency

    def get_pwm_off_target_counts(self):
        return max(0., self.target_count_delta - self.stop_lead_counts)

    def get_travelled_counts(self, count):
        if self.start_count is None or count is None:
            return None
        return max(0., count - self.start_count)

    def matches_distance(self, distance_mm, tolerance_mm):
        return math.isclose(
            abs(self.requested_distance_mm), abs(distance_mm),
            rel_tol=0., abs_tol=tolerance_mm,
        )

    def reset_standstill(self):
        self.standstill_sample_count = 0

    def observe_standstill(self, count, sample_time):
        if self.settled_count is not None:
            return True
        if self.actual_stop_print_time is None \
                or sample_time <= self.actual_stop_print_time:
            return False
        if self.last_standstill_sample_time is not None \
                and sample_time <= self.last_standstill_sample_time:
            return False

        self.last_standstill_sample_time = sample_time
        self.standstill_sample_count += 1
        if self.standstill_sample_count < self.STANDSTILL_SAMPLE_COUNT:
            return False

        self.settled_count = count
        self.settled_sample_time = sample_time
        return True


class ProcessMoveSyncMonitor:
    """Singleton monitor per MMU that owns process_move hook wrapping and move-derived sync."""

    def __init__(self, mmu):
        self.mmu = mmu
        self.hooked_extruder = self.original_process_move = None
        self.hook_enabled = False
        self.active_bldc = None

    def _attach_bldc(self, bldc):
        self.hook_enabled = True
        self.active_bldc = bldc

    def activate(self, bldc):
        if self.active_bldc is not None and self.active_bldc is not bldc:
            self.active_bldc.set_sync_enabled(False)
        if self.hook_enabled and self.active_bldc is bldc:
            return True

        unit = getattr(bldc, 'mmu_unit', None)
        extruder = (bldc.printer.lookup_object(unit.extruder_name()) if unit is not None
                    else self.mmu.toolhead.get_extruder())
        if extruder is None:
            self.mmu.log_warning("BLDC_PROCESS_MOVE: no extruder")
            return False

        process_move = getattr(extruder, 'process_move', None)
        if process_move is None:
            self.mmu.log_warning("BLDC_PROCESS_MOVE: process_move missing")
            return False

        params = list(inspect.signature(process_move).parameters)
        if params != ['print_time', 'move', 'ea_index']:
            self.mmu.log_warning("BLDC_PROCESS_MOVE: bad signature=%s" % (','.join(params),))
            return False

        owner = getattr(extruder, '_hh_bldc_process_move_owner', None)
        if owner is not None and owner is not self:
            self.mmu.log_warning("BLDC_PROCESS_MOVE: foreign hook present")
            return False

        if owner is self:
            self.hooked_extruder = extruder
            self.original_process_move = getattr(extruder, '_hh_bldc_original_process_move', process_move)
            self._attach_bldc(bldc)
            return True

        def wrapped_process_move(_hooked_self, print_time, move, ea_index):
            process_move(print_time, move, ea_index)
            self._handle_process_move(print_time, move, ea_index)

        extruder._hh_bldc_original_process_move = process_move
        extruder._hh_bldc_process_move_owner = self
        extruder.process_move = MethodType(wrapped_process_move, extruder)
        self.hooked_extruder = extruder
        self.original_process_move = process_move
        self._attach_bldc(bldc)
        return True

    def deactivate(self, bldc):
        if self.active_bldc is not bldc:
            return

        extruder = self.hooked_extruder
        if extruder is not None and getattr(extruder, '_hh_bldc_process_move_owner', None) is self \
                and self.original_process_move is not None:
            extruder.process_move = self.original_process_move
            try:
                del extruder._hh_bldc_original_process_move
                del extruder._hh_bldc_process_move_owner
            except Exception:
                pass

        self.hooked_extruder = self.original_process_move = None
        self.hook_enabled = False
        self.active_bldc = None

    def _handle_process_move(self, print_time, move, ea_index):
        bldc = self.active_bldc
        if bldc is None or not bldc.sync_active:
            return
        if ea_index < 3 or ea_index >= len(move.axes_r):
            return

        axis_r = move.axes_r[ea_index]
        if axis_r == 0.:
            return
        move_distance = move.move_d * axis_r
        if bldc._is_damped_sync_retraction(move_distance):
            return

        bldc.queue_trapzoid_move(move, axis_r, print_time, 'process_move_push')

class BldcTachometer:
    """Own tachometer sampling and PID PWM correction for BLDC."""

    TACH_VALID_MAX_AGE = 0.5
    CONTROL_DEADBAND_RPM = 50.
    CONTROL_MIN_RPM = 150.
    CONTROL_DT_MAX_SAMPLE_FACTOR = 2.
    DEFAULT_POLL_INTERVAL_S = 0.00025
    MIN_POLL_INTERVAL_S = 0.0001
    POLL_SAMPLES_PER_EDGE = 2.

    PID_PARAM_BASE = 255.
    EDGES_PER_PULSE = 2.

    def __init__(self, bldc, config):
        self.mmu = bldc.mmu
        self.section_name = bldc.section_name
        self.reactor = bldc.reactor
        self.mcu = bldc.mcu
        self.mcu_pwm_pin = bldc.mcu_pwm_pin
        self.bldc = bldc

        self.pwm_min = bldc.get_effective_pwm_min()
        self.pwm_max = bldc.pwm_max
        self.tachometer_ppr = config.getint('tachometer_ppr', 9, minval=1)
        self.tachometer_sample_time = config.getfloat('tachometer_sample_time', 0.04, above=0.)
        self.tachometer_stale_time = config.getfloat('tachometer_stale_time', self.tachometer_sample_time * 3., above=0.)
        self.control_kp = config.getfloat('tachometer_control_kp', 12.0, minval=0.) / self.PID_PARAM_BASE
        self.control_ki = config.getfloat('tachometer_control_ki', 200.0, minval=0.) / self.PID_PARAM_BASE
        self.control_max_delta_pwm = config.getfloat('tachometer_control_max_delta_pwm', 0.20, minval=0., maxval=1.)

        requested_poll_interval = config.getfloat(
            'tachometer_poll_interval', self.DEFAULT_POLL_INTERVAL_S,
            minval=self.MIN_POLL_INTERVAL_S,
        )
        max_poll_interval = self._get_poll_interval_for_rpm(
            bldc.UNCALIBRATED_MAX_RPM
        )
        self.tachometer_poll_interval = min(
            requested_poll_interval, max_poll_interval
        )

        tachometer_pin = config.get('tachometer_pin', None)
        self.tachometer = None
        if tachometer_pin is not None:
            if self.tachometer_poll_interval < requested_poll_interval:
                log_warning = self.mmu.log_warning if self.mmu is not None else logging.warning
                log_warning(
                    "BLDC tachometer_poll_interval clamped from %.6fs to "
                    "%.6fs in [%s] to prevent edge aliasing at %.1f RPM"
                    % (
                        requested_poll_interval,
                        self.tachometer_poll_interval,
                        self.section_name,
                        bldc.UNCALIBRATED_MAX_RPM,
                    )
                )
            self.tachometer = pulse_counter.MCU_counter(
                bldc.printer, tachometer_pin, self.tachometer_sample_time,
                self.tachometer_poll_interval,
            )

        self.commanded_rpm = 0.
        self.commanded_source = None
        self.last_tach_frequency = 0.
        self.last_tach_rpm = 0.
        self.last_tach_error_rpm = 0.
        self.last_tach_eventtime = None
        self.control_correction_pwm = self.integral_correction_pwm = 0.
        self.control_reason = 'disabled'
        self._tach_last_count = None
        self._tach_last_count_time = None
        self.enabled = False

        if self.tachometer is not None:
            self.tachometer.setup_callback(self.handle_tachometer)

    def has_tachometer(self):
        return self.tachometer is not None

    def get_count_sample(self):
        return self._tach_last_count, self._tach_last_count_time

    def get_counts_per_revolution(self):
        return self.EDGES_PER_PULSE * self.tachometer_ppr

    def _get_poll_interval_for_rpm(self, rpm):
        return 60. / (
            rpm
            * self.get_counts_per_revolution()
            * self.POLL_SAMPLES_PER_EDGE
        )

    def get_max_reliable_rpm(self):
        if self.tachometer is None:
            return INFINITY
        return 60. / (
            self.tachometer_poll_interval
            * self.get_counts_per_revolution()
            * self.POLL_SAMPLES_PER_EDGE
        )

    def _get_pid_skip_reason(self):
        if self.tachometer is None:
            return 'no_tachometer'
        if not self.enabled:
            return 'pid_disabled'
        if self.commanded_rpm <= EPSILON or self.commanded_source == 'stop':
            return 'stopped'
        if self.commanded_rpm < self.CONTROL_MIN_RPM:
            return 'below_min_rpm'
        if not self.has_fresh_tachometer():
            return 'tach_stale'
        return None

    def set_commanded(self, rpm, source):
        self.commanded_rpm = rpm
        self.commanded_source = source

    def stop(self):
        self.commanded_rpm = 0.
        self.commanded_source = 'stop'
        self.integral_correction_pwm = 0.
        self.set_control_state('stopped', 0., 0.)

    def reset_integral(self):
        self.integral_correction_pwm = 0.
        self.control_correction_pwm = 0.

    def apply_control(self, pwm):
        if pwm <= EPSILON:
            self.set_control_state('zero_pwm', 0., 0.)
            return 0.

        reason = self._get_pid_skip_reason()
        if reason is not None:
            self.set_control_state(reason, 0.)
            return pwm

        applied_pwm = min(self.pwm_max, max(self.pwm_min, pwm + self.control_correction_pwm))
        self.set_control_state('active', self.control_correction_pwm, self.last_tach_error_rpm)
        return applied_pwm

    def get_current_print_time(self):
        return self.mcu.estimated_print_time(self.reactor.monotonic())

    def has_fresh_tachometer(self):
        if self.last_tach_eventtime is None:
            return False
        current_print_time = self.get_current_print_time()
        return current_print_time - self.last_tach_eventtime <= self.tachometer_stale_time

    def is_tach_sample_valid(self):
        if self.tachometer is None:
            return False, 'no_tachometer'
        if self.last_tach_eventtime is None:
            return False, 'tach_missing'
        if self.get_current_print_time() - self.last_tach_eventtime > self.TACH_VALID_MAX_AGE:
            return False, 'tach_stale'
        if self.last_tach_rpm <= EPSILON:
            return False, 'tach_zero'
        return True, 'ok'

    def get_status(self):
        has_tachometer = self.has_tachometer()
        return {
            'tachometer_rpm': self.last_tach_rpm,
            'tachometer_error_rpm': self.commanded_rpm - self.last_tach_rpm,
            'control_enabled': has_tachometer and self.enabled,
            'control_reason': self.control_reason,
            'control_correction_pwm': self.control_correction_pwm,
            'integral_correction_pwm': self.integral_correction_pwm,
        }

    def set_control_state(self, reason, correction_pwm=0., error_rpm=None):
        self.control_reason = reason
        self.control_correction_pwm = correction_pwm
        if error_rpm is not None:
            self.last_tach_error_rpm = error_rpm

    def _log_control_state(self, print_time, control_dt):
        self.mmu.log_stepper(
            "BLDC_CONTROL: reason=%s commanded_rpm=%.1f tach_rpm=%.1f error_rpm=%.1f "
            "correction_pwm=%.4f integral_pwm=%.4f control_dt=%.4f source=%s "
            "print_time=%.6f unit=%s"
            % (
                self.control_reason, self.commanded_rpm, self.last_tach_rpm,
                self.last_tach_error_rpm, self.control_correction_pwm,
                self.integral_correction_pwm, control_dt, self.commanded_source,
                print_time, self.section_name,
            )
        )

    def handle_tachometer(self, time, count, count_time):
        if self._tach_last_count is None:
            self._tach_last_count = count
            self._tach_last_count_time = count_time
            self.last_tach_eventtime = time
            self.bldc.handle_tachometer_position_sample(time, count, count_time)
            return
        delta_time = count_time - self._tach_last_count_time
        frequency = (count - self._tach_last_count) / delta_time if delta_time > 0. else 0.
        control_dt = delta_time if delta_time > 0. else (max(0., time - self.last_tach_eventtime) if self.last_tach_eventtime is not None else 0.)
        if control_dt <= 0.:
            control_dt = self.tachometer_sample_time
        control_dt = min(control_dt, self.tachometer_sample_time * self.CONTROL_DT_MAX_SAMPLE_FACTOR)
        self._tach_last_count = count
        self._tach_last_count_time = count_time
        self.last_tach_frequency = frequency
        self.last_tach_eventtime = time
        tach_rpm = frequency * 30. / self.tachometer_ppr

        if abs(self.last_tach_rpm - tach_rpm) > EPSILON:
            self.mmu.log_stepper(
                "BLDC_TACH: freq=%.4f rpm=%.1f print_time=%.6f unit=%s"
                % (frequency, tach_rpm, time, self.section_name)
            )

        self.last_tach_rpm = tach_rpm
        error_rpm = self.commanded_rpm - self.last_tach_rpm

        reason = self._get_pid_skip_reason()
        if reason is not None:
            self.integral_correction_pwm = 0.
            self.set_control_state(reason, 0., error_rpm)
        elif abs(error_rpm) <= self.CONTROL_DEADBAND_RPM:
            self.set_control_state('deadband', self.control_correction_pwm, error_rpm)
        else:
            c = self.control_max_delta_pwm
            norm = (error_rpm / self.bldc.get_effective_max_rpm()) * (self.pwm_max - self.pwm_min)
            p_correction = max(-c, min(c, self.control_kp * norm))
            self.integral_correction_pwm = max(-c, min(c, self.integral_correction_pwm + self.control_ki * norm * control_dt))
            total_correction = max(-c, min(c, p_correction + self.integral_correction_pwm))
            self.set_control_state('active', total_correction, error_rpm)
        if self.control_reason != 'pid_disabled':
            self._log_control_state(time, control_dt)
        self.bldc.handle_tachometer_position_sample(time, count, count_time)

class MmuGearBldc:
    """BLDC gear controller for MMU gear motion replacement. Designed to be used with a BLDC motor and external ESC for gear drive sections, with optional tachometer feedback and closed loop control."""

    CALIBRATION_DEFAULT_POINTS = 16
    CALIBRATION_DEFAULT_SAMPLE_S = 1.0
    CALIBRATION_EXTENDED_SAMPLE_S = 3.0
    CALIBRATION_DEFAULT_SETTLE_S = 0.35
    CALIBRATION_STANDSTILL_TIMEOUT_S = 3.0
    CALIBRATION_STANDSTILL_SAMPLES = 2
    CALIBRATION_MIN_SAMPLES = 3
    CALIBRATION_MIN_POINTS = 3
    CALIBRATION_DISCOVERY_MAX_STEP = 0.05
    CALIBRATION_STARTUP_PWM_OFFSET = 0.03
    CALIBRATION_PWM_QUANTUM = 0.01
    UNCALIBRATED_MAX_RPM = 5500.
    LARGE_SPEED_CHANGE_RPM = 500.
    PWM_WRITE_MIN_INTERVAL_S = 0.1
    PWM_WRITE_MIN_DELTA = 0.002
    BRAKE_MIN_TIME_S = 0.03
    BRAKE_MIN_ACTIVE_PWM = 0.08
    PIN_SCHEDULE_LEAD_S = 0.25
    POSITION_NO_PROGRESS_TIMEOUT_S = 1.0
    POSITION_STANDSTILL_TIMEOUT_S = 1.0
    POSITION_DISTANCE_TOLERANCE_MM = 0.01
    POSITION_WAIT_POLL_MAX_S = 0.05
    POSITION_FREQUENCY_FILTER_WEIGHT = 0.5
    POSITION_LEARNING_WEIGHT = 0.5
    POSITION_INITIAL_STOP_LEAD_COUNTS = 1.0
    POSITION_MAX_STOP_LEAD_REVOLUTIONS = 2.0
    POSITION_MAX_STOP_LEAD_RATIO = 0.25

    MOTION_STATE_IDLE = 'idle'
    MOTION_STATE_MOVING = 'moving'
    MOTION_STATE_STOP = 'stop'
    MOTION_STATE_BRAKE = 'brake'

    def __init__(self, config, mmu, first_gate=0, num_gates=1):
        self.config = config
        self.mmu = mmu
        self.first_gate = first_gate
        self.num_gates = num_gates
        self.printer = config.get_printer()
        self.reactor = self.printer.get_reactor()

        self.sync_active = False
        self.motion_state = self.MOTION_STATE_IDLE
        self.motion_queue = []
        self.motion_timer = None
        self._motion_timer_running = False

        self.last_pwm = self.last_effective_pwm = 0.
        self._last_pwm_write_print_time = 0.
        self._last_dir_write_print_time = 0.
        self.last_dir = None
        self.section_name = config.get_name()
        self.commanded_rpm = self.commanded_linear_speed = 0.
        self.commanded_source = None
        self._finite_move_tracker = None
        self._position_feed_forward = {
            direction: PositionFeedForwardState(
                self.POSITION_INITIAL_STOP_LEAD_COUNTS,
                self.POSITION_LEARNING_WEIGHT,
            )
            for direction in (-1, 1)
        }
        self.calibration_map_points = []
        self.map_mode = 'linear'
        self.map_fallback_reason = 'map_missing'

        self.kick_start_time = config.getfloat('kick_start_time', 0.05, minval=0.)
        self.brake_pwm = config.getfloat('brake_pwm', 1., minval=0., maxval=1.)
        self.brake_max_time = config.getfloat('brake_max_time', 0.25, minval=0.)

        self.pwm_min = config.getfloat('pwm_min', 0.2, minval=0., maxval=1.)
        self.calibrated_pwm_min = None
        self.pwm_max = config.getfloat('pwm_max', 1.0, minval=0., maxval=1.)
        if self.pwm_min > self.pwm_max:
            raise config.error("'pwm_min' cannot be greater than 'pwm_max' in [%s]" % config.get_name())
        self.torque_assist_pwm = config.getfloat(
            'torque_assist_pwm', self.pwm_min, minval=0., maxval=1.
        )
        if self.torque_assist_pwm > self.pwm_max:
            raise config.error(
                "'torque_assist_pwm' cannot be greater than 'pwm_max' in [%s]"
                % config.get_name()
            )
        self.sync_retract_min_distance = config.getfloat(
            'sync_retract_min_distance', 1., minval=0.
        )

        self.rotation_distance = config.getfloat('rotation_distance', 1.0, above=0.)
        self.direction_map = self._load_direction_map(config)

        self.hardware_pwm = config.getboolean('hardware_pwm', True)
        self.cycle_time = config.getfloat('cycle_time', 0.00005, above=0.)

        ppins = self.printer.lookup_object('pins')

        self.mcu_dir_pin = ppins.setup_pin('digital_out', config.get('dir_pin'))
        self.mcu_dir_pin.setup_max_duration(0.)
        self.mcu_dir_pin.setup_start_value(0., 0.)

        pwm_pin_params = ppins.lookup_pin(config.get('pwm_pin'), can_invert=True)
        if 'config' in inspect.signature(MCU_queued_pwm.__init__).parameters:
            self.mcu_pwm_pin = MCU_queued_pwm(config, pwm_pin_params)
        else:
            self.mcu_pwm_pin = MCU_queued_pwm(pwm_pin_params)
        self.mcu_pwm_pin.setup_max_duration(0.)
        self.mcu_pwm_pin.setup_cycle_time(self.cycle_time, self.hardware_pwm)
        self.mcu_pwm_pin.setup_start_value(0., 0.)

        self.mcu = self.mcu_pwm_pin.get_mcu()

        if self.mcu is not self.mcu_dir_pin.get_mcu():
            raise config.error(
                "'pwm_pin' and 'dir_pin' in [%s] must be on the same mcu"
                % config.get_name()
            )

        self.tachometer = BldcTachometer(self, config)
        self.min_schedule_time = (
            self.mcu.min_schedule_time() + self.PIN_SCHEDULE_LEAD_S
        )
        self.direction_setup_time = self.mcu.min_schedule_time()
        self.motion_sample_time = min(
            self.mcu.min_schedule_time(),
            self.tachometer.tachometer_sample_time,
        )
        self.pin_request_queue = BldcPinRequestQueue.get_for_mcu(
            config, self.mcu
        )

        self.active_sync_monitor = None
        self.printer.register_event_handler('klippy:connect', self._handle_connect)
        self.printer.register_event_handler('klippy:shutdown', self._handle_shutdown)

    def _load_direction_map(self, config):
        default = [gate % 2 for gate in range(self.num_gates)]
        values = list(config.getintlist('direction_map', default))
        if len(values) != self.num_gates:
            raise config.error(
                "'direction_map' in [%s] must contain exactly %d entries (one per unit-local gate)"
                % (config.get_name(), self.num_gates)
            )
        for idx, value in enumerate(values):
            if value not in (0, 1):
                raise config.error(
                    "'direction_map' in [%s] has invalid value %s at index %d (allowed: 0 or 1)"
                    % (config.get_name(), value, idx)
                )
        return values

    def _map_distance_for_gate(self, requested_dist, gate=None):
        gate = self.mmu.gate_selected if gate is None else gate
        map_value = 0 if gate is None or not self.supports_gate(gate) else self.direction_map[gate - self.first_gate]
        return (-requested_dist if map_value else requested_dist), map_value, gate

    def _map_forward_for_gate(self, requested_forward, requested_speed):
        return self._map_distance_for_gate(
            abs(requested_speed) if requested_forward else -abs(requested_speed)
        )[0] >= 0.



    def _get_calibrated_max_rpm(self):
        return self.calibration_map_points[-1]['rpm'] if self.calibration_map_points else None

    def _get_calibrated_min_rpm(self):
        return self.calibration_map_points[0]['rpm'] if self.calibration_map_points else None

    def get_effective_pwm_min(self):
        return self.calibrated_pwm_min if self.calibrated_pwm_min is not None else self.pwm_min

    def _invalidate_calibration(self, reason):
        self.calibration_map_points = []
        self.calibrated_pwm_min = None
        self.tachometer.pwm_min = self.pwm_min
        self.map_mode, self.map_fallback_reason = 'linear', reason
        return False, reason

    def get_effective_max_rpm(self):
        configured_max_rpm = (
            self._get_calibrated_max_rpm() or self.UNCALIBRATED_MAX_RPM
        )
        return min(
            configured_max_rpm,
            self.tachometer.get_max_reliable_rpm(),
        )

    def get_effective_max_linear_speed(self):
        return self.get_effective_max_rpm() * self.rotation_distance / 60.

    def get_effective_min_linear_speed(self):
        if not self.tachometer.has_tachometer():
            return 0.
        min_rpm = self._get_calibrated_min_rpm()
        return min_rpm * self.rotation_distance / 60. if min_rpm is not None else 0.

    def _is_torque_assist_speed(self, speed_mm_s):
        minimum_speed = self.get_effective_min_linear_speed()
        return (
            self.torque_assist_pwm > EPSILON
            and minimum_speed > EPSILON
            and EPSILON < abs(speed_mm_s) < minimum_speed - EPSILON
        )

    def _is_damped_sync_retraction(self, move_distance):
        return (
            move_distance < -EPSILON
            and -move_distance < self.sync_retract_min_distance
        )

    def _get_achievable_linear_speed(self, requested_speed):
        min_speed = self.get_effective_min_linear_speed()
        max_speed = self.get_effective_max_linear_speed()
        return min(max(requested_speed, min_speed), max_speed)

    def _normalize_calibration_points(self, raw_points):
        if not isinstance(raw_points, list):
            return None, 'points_type'

        pairs = []
        for point in raw_points:
            pwm = float(point.get('pwm'))
            rpm = float(point.get('rpm'))
            if pwm < self.pwm_min - EPSILON or pwm > self.pwm_max + EPSILON:
                return None, 'pwm_range'
            if rpm <= EPSILON:
                return None, 'rpm_range'
            pairs.append((pwm, rpm))

        if not pairs:
            return None, 'points_empty'

        pwm_buckets = {}
        for pwm, rpm in pairs:
            pwm_buckets.setdefault(round(pwm, 6), []).append(rpm)

        normalized = []
        discarded_count = 0
        for pwm_key in sorted(pwm_buckets.keys()):
            avg_rpm = sum(pwm_buckets[pwm_key]) / len(pwm_buckets[pwm_key])
            point = {'pwm': pwm_key, 'rpm': round(avg_rpm, 1)}
            if normalized and point['rpm'] <= normalized[-1]['rpm'] + EPSILON:
                discarded_count += 1
                continue
            normalized.append(point)

        if len(normalized) < self.CALIBRATION_MIN_POINTS:
            return None, 'insufficient_points'

        if discarded_count:
            self.mmu.log_warning(
                "Ignored %d non-increasing BLDC calibration point%s in [%s]"
                % (
                    discarded_count,
                    "" if discarded_count == 1 else "s",
                    self.section_name,
                )
            )

        return normalized, None

    def set_calibration_map(self, payload):
        if payload is None:
            return self._invalidate_calibration('map_missing')

        if not isinstance(payload, dict):
            return self._invalidate_calibration('payload_type')

        if set(payload) != {'pwm_min', 'points'}:
            return self._invalidate_calibration('payload_schema')
        try:
            calibrated_pwm_min = float(payload['pwm_min'])
            raw_points = payload['points']
        except (TypeError, ValueError):
            return self._invalidate_calibration('payload_type')
        if not math.isfinite(calibrated_pwm_min):
            return self._invalidate_calibration('pwm_min_type')
        if calibrated_pwm_min < self.pwm_min - EPSILON or calibrated_pwm_min > self.pwm_max + EPSILON:
            return self._invalidate_calibration('pwm_min_range')
        if not isinstance(raw_points, list) or len(raw_points) < self.CALIBRATION_MIN_POINTS:
            return self._invalidate_calibration('points_type')

        points = []
        previous_pwm = previous_rpm = None
        for point in raw_points:
            if not isinstance(point, dict) or set(point) != {'pwm', 'rpm'}:
                return self._invalidate_calibration('point_schema')
            try:
                pwm = float(point['pwm'])
                rpm = float(point['rpm'])
            except (TypeError, ValueError):
                return self._invalidate_calibration('point_type')
            if not math.isfinite(pwm) or not math.isfinite(rpm):
                return self._invalidate_calibration('point_type')
            if pwm < self.pwm_min - EPSILON or pwm > self.pwm_max + EPSILON:
                return self._invalidate_calibration('pwm_range')
            if rpm <= EPSILON:
                return self._invalidate_calibration('rpm_range')
            if previous_pwm is not None and pwm <= previous_pwm + EPSILON:
                return self._invalidate_calibration('pwm_order')
            if previous_rpm is not None and rpm <= previous_rpm + EPSILON:
                return self._invalidate_calibration('rpm_order')
            points.append({'pwm': round(pwm, 6), 'rpm': round(rpm, 1)})
            previous_pwm, previous_rpm = pwm, rpm

        if abs(points[0]['pwm'] - calibrated_pwm_min) > EPSILON:
            return self._invalidate_calibration('pwm_min_mismatch')

        self.calibrated_pwm_min = round(calibrated_pwm_min, 6)
        self.calibration_map_points = points
        self.tachometer.pwm_min = self.get_effective_pwm_min()
        self.map_mode, self.map_fallback_reason = 'mapped', 'none'
        return True, None

    def get_calibration_map_payload(self):
        if not self.calibration_map_points:
            return None
        return {
            'pwm_min': self.get_effective_pwm_min(),
            'points': [dict(point) for point in self.calibration_map_points],
        }

    def rpm_to_pwm(self, rpm):
        if self.calibration_map_points and self.tachometer.has_tachometer():
            self.map_mode, self.map_fallback_reason = 'mapped', 'none'
            return self._rpm_to_pwm_mapped(rpm)
        reason = 'map_missing' if not self.calibration_map_points else 'tachometer_missing'
        self.map_mode = 'linear'
        self.map_fallback_reason = reason
        if rpm <= EPSILON:
            return 0.
        effective_pwm_min = self.get_effective_pwm_min()
        return effective_pwm_min + (self.pwm_max - effective_pwm_min) * min(rpm / self.get_effective_max_rpm(), 1.)

    def _rpm_to_pwm_mapped(self, rpm):
        points = self.calibration_map_points
        min_point = min(points, key=lambda p: p['rpm'])
        max_point = max(points, key=lambda p: p['rpm'])
        if rpm <= min_point['rpm']:
            return min_point['pwm']
        if rpm >= max_point['rpm']:
            return max_point['pwm']
        for lower, upper in zip(points, points[1:]):
            if lower['rpm'] <= rpm <= upper['rpm']:
                fraction = (rpm - lower['rpm']) / (upper['rpm'] - lower['rpm'])
                return lower['pwm'] + fraction * (upper['pwm'] - lower['pwm'])
        return points[-1]['pwm']

    def _build_calibration_sweep(self, start_pwm, point_count):
        if point_count <= 1:
            return [self.pwm_max]
        step = (self.pwm_max - start_pwm) / float(point_count - 1)
        return [start_pwm + (step * i) for i in range(point_count)]

    def _apply_startup_pwm_margin(self, startup_pwm):
        """Add fixed startup headroom and round up to PWM resolution."""
        margin_pwm = min(self.pwm_max, startup_pwm + self.CALIBRATION_STARTUP_PWM_OFFSET)
        rounded_pwm = math.ceil(
            margin_pwm / self.CALIBRATION_PWM_QUANTUM - EPSILON
        ) * self.CALIBRATION_PWM_QUANTUM
        return min(self.pwm_max, rounded_pwm)

    def _sample_first_map_point(self, pwm):
        """Sample first map point with extended settling for marginal startup."""
        return self._sample_calibration_window(pwm, EPSILON)

    def _calibration_sample_is_fresh(self, previous_eventtime):
        eventtime = self.tachometer.last_tach_eventtime
        if eventtime is None:
            return False
        return previous_eventtime is None or eventtime > previous_eventtime + EPSILON

    def _calibration_uses_injected_tachometer(self):
        validator = self.tachometer.is_tach_sample_valid
        return getattr(validator, '__func__', None) is not BldcTachometer.is_tach_sample_valid

    def _sample_calibration_window(self, pwm, previous_positive_rpm=None):
        self._safe_set_pwm(pwm)
        self.mmu.movequeue_wait()
        deadline = self.reactor.monotonic() + self.CALIBRATION_DEFAULT_SETTLE_S
        self.reactor.pause(deadline)

        readings = []
        previous_eventtime = None
        injected_tachometer = self._calibration_uses_injected_tachometer()
        sample_start = self.reactor.monotonic()
        sample_deadline = sample_start + self.CALIBRATION_DEFAULT_SAMPLE_S
        extended_deadline = sample_start + self.CALIBRATION_EXTENDED_SAMPLE_S
        while self.reactor.monotonic() < sample_deadline:
            valid, _reason = self.tachometer.is_tach_sample_valid()
            is_fresh = self._calibration_sample_is_fresh(previous_eventtime)
            if injected_tachometer and self.tachometer.last_tach_eventtime is None:
                is_fresh = True
            has_fresh_tachometer = injected_tachometer or self.tachometer.has_fresh_tachometer()
            if is_fresh and has_fresh_tachometer and (
                valid or self.tachometer.last_tach_rpm <= EPSILON
            ):
                readings.append(self.tachometer.last_tach_rpm)
                previous_eventtime = self.tachometer.last_tach_eventtime
            self.reactor.pause(min(
                sample_deadline,
                self.reactor.monotonic() + self.tachometer.tachometer_sample_time,
            ))

        median_rpm = self._calibration_median(readings)
        needs_extended_sampling = (
            previous_positive_rpm is not None
            and (
                median_rpm <= EPSILON
                or median_rpm < previous_positive_rpm - EPSILON
            )
        )
        while needs_extended_sampling and self.reactor.monotonic() < extended_deadline:
            valid, _reason = self.tachometer.is_tach_sample_valid()
            is_fresh = self._calibration_sample_is_fresh(previous_eventtime)
            if injected_tachometer and self.tachometer.last_tach_eventtime is None:
                is_fresh = True
            has_fresh_tachometer = injected_tachometer or self.tachometer.has_fresh_tachometer()
            if is_fresh and has_fresh_tachometer and (
                valid or self.tachometer.last_tach_rpm <= EPSILON
            ):
                readings.append(self.tachometer.last_tach_rpm)
                previous_eventtime = self.tachometer.last_tach_eventtime
            self.reactor.pause(min(
                extended_deadline,
                self.reactor.monotonic() + self.tachometer.tachometer_sample_time,
            ))

        median_rpm = self._calibration_median(readings)
        if len(readings) < self.CALIBRATION_MIN_SAMPLES:
            raise MmuError(
                "BLDC calibration failed for [%s]: insufficient fresh tachometer readings (%d/%d)"
                % (self.section_name, len(readings), self.CALIBRATION_MIN_SAMPLES)
            )
        return {'pwm': round(pwm, 6), 'rpm': round(median_rpm, 1)}

    @staticmethod
    def _calibration_median(readings):
        if not readings:
            raise MmuError("BLDC calibration failed: no tachometer readings")
        ordered_readings = sorted(readings)
        middle = len(ordered_readings) // 2
        if len(ordered_readings) % 2:
            return ordered_readings[middle]
        return (ordered_readings[middle - 1] + ordered_readings[middle]) / 2.

    def _wait_for_calibration_standstill(self):
        self.stop()
        self.mmu.movequeue_wait()
        deadline = self.reactor.monotonic() + self.CALIBRATION_STANDSTILL_TIMEOUT_S
        zero_samples = 0
        previous_eventtime = None
        injected_tachometer = self._calibration_uses_injected_tachometer()
        if injected_tachometer:
            self.tachometer.last_tach_rpm = 0.
        while self.reactor.monotonic() < deadline:
            eventtime = self.tachometer.last_tach_eventtime
            if (
                (self._calibration_sample_is_fresh(previous_eventtime) or injected_tachometer)
                and self.tachometer.last_tach_rpm <= EPSILON
            ):
                zero_samples += 1
                previous_eventtime = eventtime
                if zero_samples >= self.CALIBRATION_STANDSTILL_SAMPLES:
                    return
            else:
                zero_samples = 0
            self.reactor.pause(self.reactor.monotonic() + self.tachometer.tachometer_sample_time)
        raise MmuError(
            "BLDC calibration failed for [%s]: tachometer did not reach standstill"
            % self.section_name
        )

    def _discover_startup_pwm(self, point_count):
        pwm_range = self.pwm_max - self.pwm_min
        discovery_count = max(
            point_count,
            int(math.ceil(pwm_range / self.CALIBRATION_DISCOVERY_MAX_STEP)) + 1,
        )
        previous_pwm = self.pwm_min
        self._wait_for_calibration_standstill()
        first_point = self._sample_calibration_window(previous_pwm)
        if first_point['rpm'] > EPSILON:
            return first_point['pwm']

        for pwm in self._build_calibration_sweep(self.pwm_min, discovery_count)[1:]:
            self._wait_for_calibration_standstill()
            point = self._sample_calibration_window(pwm)
            if point['rpm'] > EPSILON:
                moving_pwm = point['pwm']
                while moving_pwm - previous_pwm > 0.01:
                    midpoint = (previous_pwm + moving_pwm) / 2.
                    self._wait_for_calibration_standstill()
                    midpoint_point = self._sample_calibration_window(midpoint)
                    if midpoint_point['rpm'] > EPSILON:
                        moving_pwm = midpoint
                    else:
                        previous_pwm = midpoint
                if moving_pwm >= self.pwm_max - EPSILON and point_count > 1:
                    raise MmuError(
                        "BLDC calibration failed for [%s]: motor only starts at pwm_max"
                        % self.section_name
                    )
                return round(moving_pwm, 6)
            previous_pwm = pwm

        raise MmuError(
            "BLDC calibration failed for [%s]: no positive RPM detected"
            % self.section_name
        )

    def _aggregate_calibration_passes(self, pass_points):
        aggregated_points = []
        for point_index in range(len(pass_points[0])):
            readings = [points[point_index]['rpm'] for points in pass_points]
            aggregated_points.append({
                'pwm': pass_points[0][point_index]['pwm'],
                'rpm': round(self._calibration_median(readings), 1),
            })
        return aggregated_points

    def _is_calibration_rpm_drop(self, rpm, previous_rpm):
        return previous_rpm is not None and rpm < previous_rpm - EPSILON

    def _format_calibration_sweep_points(self, pass_index, passes, points):
        lines = [
            "BLDC calibration map pass %d/%d:" % (pass_index, passes),
            "+------+----------+----------+----------+",
            "| Pt   | PWM      | RPM      | Status   |",
            "+------+----------+----------+----------+",
        ]
        previous_rpm = None
        for point_index, point in enumerate(points, start=1):
            rpm = point['rpm']
            status = 'ERROR' if self._is_calibration_rpm_drop(
                rpm, previous_rpm
            ) else 'OK'
            lines.append(
                "| %4d | %8.3f | %8.1f | %8s |"
                % (point_index, point['pwm'], rpm, status)
            )
            previous_rpm = rpm
        lines.append("+------+----------+----------+----------+")
        return "\n".join(lines)

    def calibrate_pwm_rpm_map(self, point_count, passes=1, progress_callback=None):
        if not self.tachometer.has_tachometer():
            raise MmuError("BLDC tachometer unavailable for calibration in [%s]" % self.section_name)
        if not 1 <= passes <= 10:
            raise MmuError("BLDC calibration failed for [%s]: passes must be between 1 and 10" % self.section_name)

        pass_points = []
        saved_enabled = self.tachometer.enabled
        saved_kick_start_time = self.kick_start_time
        self._finite_move_tracker = None
        self.tachometer.enabled = False
        self.kick_start_time = 0.
        try:
            self.stop()
            self.mmu.movequeue_wait()
            self._safe_set_direction(True)

            startup_floors = []
            for pass_index in range(passes):
                startup_pwm = self._discover_startup_pwm(point_count)
                startup_floors.append(startup_pwm)
                if progress_callback is not None:
                    progress_callback(
                        "BLDC discovery pass %d/%d: pwm_min=%.3f"
                        % (pass_index + 1, passes, startup_pwm)
                    )

            effective_pwm_min = self._apply_startup_pwm_margin(max(startup_floors))
            map_sweep = self._build_calibration_sweep(effective_pwm_min, point_count)
            has_rpm_drop = False
            for pass_index in range(passes):
                self._wait_for_calibration_standstill()
                current_points = []
                for pwm in map_sweep:
                    previous_rpm = current_points[-1]['rpm'] if current_points else None
                    point = (
                        self._sample_first_map_point(pwm)
                        if previous_rpm is None
                        else self._sample_calibration_window(pwm, previous_rpm)
                    )
                    if point['rpm'] <= EPSILON:
                        raise MmuError(
                            "BLDC calibration failed for [%s]: zero RPM at pwm=%.4f"
                            % (self.section_name, pwm)
                    )
                    current_points.append(point)
                    if self._is_calibration_rpm_drop(point['rpm'], previous_rpm):
                        has_rpm_drop = True
                pass_points.append(current_points)
                if progress_callback is not None:
                    progress_callback(
                        self._format_calibration_sweep_points(
                            pass_index + 1, passes, current_points
                        )
                    )

            if has_rpm_drop:
                raise MmuError(
                    "BLDC calibration failed for [%s]: RPM decreased as PWM increased. "
                    "Check tachometer_ppr configuration; value may be too high."
                    % self.section_name
                )

            raw_points = self._aggregate_calibration_passes(pass_points)
        finally:
            self.tachometer.enabled = saved_enabled
            self.kick_start_time = saved_kick_start_time
            self.stop()
            self.mmu.movequeue_wait()

        points, reason = self._normalize_calibration_points(raw_points)
        if reason is not None:
            raise MmuError(
                "BLDC calibration failed for [%s]: %s (valid points=%d)"
                % (self.section_name, reason, len(raw_points))
            )
        ok, reason = self.set_calibration_map({'pwm_min': points[0]['pwm'], 'points': points})
        if not ok:
            raise MmuError(
                "BLDC calibration failed for [%s]: %s (valid points=%d)"
                % (self.section_name, reason, len(raw_points))
            )
        payload = self.get_calibration_map_payload()
        if progress_callback is not None:
            discarded_count = len(raw_points) - len(points)
            progress_callback(
                "BLDC calibration complete: pwm_min=%.3f, discarded=%d"
                % (payload['pwm_min'], discarded_count)
            )
        return payload

    def supports_gate(self, gate):
        return gate is not None and self.first_gate <= gate < self.first_gate + self.num_gates

    def _get_pin_name(self, mcu_pin):
        return {
            self.mcu_pwm_pin: 'pwm',
            self.mcu_dir_pin: 'dir',
        }.get(mcu_pin, 'unknown')

    def _get_queue_delay(self, print_time, pin_updates):
        for mcu_pin, value in pin_updates:
            if mcu_pin is not self.mcu_pwm_pin or value <= EPSILON \
                    or abs(value - self.last_pwm) < self.PWM_WRITE_MIN_DELTA:
                continue
            if self._is_endpoint_reserved(print_time):
                continue
            if self._last_pwm_write_print_time <= EPSILON:
                continue
            elapsed = print_time - self._last_pwm_write_print_time
            if elapsed >= self.PWM_WRITE_MIN_INTERVAL_S - EPSILON:
                continue
            wait_s = self.PWM_WRITE_MIN_INTERVAL_S - elapsed
            self.mmu.log_stepper(
                "BLDC_SET_PIN: delay pin=pwm value=%.4f wait=%.6f print_time=%.6f"
                % (value, wait_s, print_time)
            )
            if self.pin_request_queue.native_mode:
                return 'reschedule', print_time + wait_s
            return 'delay', wait_s
        return None

    def queue_pin_updates(self, print_time, pin_updates):
        delay = self._get_queue_delay(print_time, pin_updates)
        if delay is not None:
            return delay

        applied = False
        for mcu_pin, value in pin_updates:
            pin_name = self._get_pin_name(mcu_pin)
            if mcu_pin is self.mcu_pwm_pin:
                if value > EPSILON and self._is_endpoint_reserved(print_time):
                    self.mmu.log_stepper(
                        "BLDC_SET_PIN: discard pin=pwm value=%.4f print_time=%.6f reason=endpoint_reserve"
                        % (value, print_time)
                    )
                    continue
                if abs(value - self.last_pwm) < self.PWM_WRITE_MIN_DELTA:
                    self.mmu.log_stepper(
                        "BLDC_SET_PIN: discard pin=pwm value=%.4f print_time=%.6f reason=small_delta"
                        % (value, print_time)
                    )
                    continue
                applied = self._set_pwm_callback(print_time, value) != 'discard' or applied
                if value > EPSILON:
                    self._last_pwm_write_print_time = print_time
            elif mcu_pin is self.mcu_dir_pin:
                applied = self._set_dir_callback(print_time, value) != 'discard' or applied
            else:
                self.mmu.log_stepper(
                    "BLDC_SET_PIN: discard pin=%s value=%.4f print_time=%.6f reason=bad_pin"
                    % (pin_name, value, print_time)
                )

        if not applied:
            return 'discard', 0.
        return None

    def _set_pwm_callback(self, print_time, value):
        if abs(value - self.last_pwm) < EPSILON:
            return 'discard'
        self.mcu_pwm_pin.set_pwm(print_time, value)
        self.last_pwm = value
        tracker = self._finite_move_tracker
        if value <= EPSILON and tracker is not None \
                and tracker.stop_dispatched and not tracker.aborted:
            tracker.actual_stop_print_time = print_time
        self.mmu.log_stepper("BLDC_SET_PIN: pwm value=%.4f print_time=%.6f" % (value, print_time))
        return ''

    def _set_dir_callback(self, print_time, value):
        ivalue = 1 if value else 0
        if self.last_dir == ivalue:
            return 'discard'
        self.mcu_dir_pin.set_digital(print_time, ivalue)
        self.last_dir = ivalue
        self._last_dir_write_print_time = print_time
        self.mmu.log_stepper("BLDC_SET_PIN: dir value=%d print_time=%.6f" % (ivalue, print_time))
        return ''

    def _send_pin_updates(self, pin_updates, print_time):
        t = self._get_scheduled_print_time() + self.min_schedule_time
        floored_print_time = t if print_time is None or print_time < t else print_time
        valid_updates = tuple(
            (mcu_pin, value) for mcu_pin, value in pin_updates
            if mcu_pin is self.mcu_pwm_pin or mcu_pin is self.mcu_dir_pin
        )
        if valid_updates:
            self.pin_request_queue.send(
                (self.queue_pin_updates, valid_updates), floored_print_time
            )

    def _send_pin(self, mcu_pin, value, print_time):
        self._send_pin_updates(((mcu_pin, value),), print_time)

    def _safe_set_direction(self, forward, print_time=None):
        self._send_pin(self.mcu_dir_pin, int(forward), print_time)

    def _safe_set_pwm(self, value, print_time=None):
        # Queue callback applies write-throttle policy to unify native/fallback behavior.
        self._send_pin(self.mcu_pwm_pin, value, print_time)

    def _queue_pwm_if_changed(self, value, print_time):
        if value <= EPSILON:
            value = 0.
            if self.last_effective_pwm <= EPSILON and self.last_pwm <= EPSILON:
                return False
        elif abs(value - self.last_effective_pwm) < self.PWM_WRITE_MIN_DELTA:
            return False

        self.last_effective_pwm = value
        self._send_pin(self.mcu_pwm_pin, value, print_time)
        return True

    def _is_endpoint_reserved(self, print_time):
        tracker = self._finite_move_tracker
        if tracker is None or tracker.endpoint_locked \
                or tracker.stop_descriptor is None:
            return False
        reservation_time = tracker.stop_descriptor.print_time
        if tracker.nominal_target_print_time is not None \
                and print_time < tracker.nominal_target_print_time:
            reservation_time = min(
                reservation_time, tracker.nominal_target_print_time
            )
        return reservation_time - print_time \
            < self.mcu.min_schedule_time() - EPSILON

    def _queue_drive_if_changed(self, direction, value, print_time):
        pin_updates = [(self.mcu_dir_pin, direction)]
        if value <= EPSILON:
            value = 0.
            if self.last_effective_pwm > EPSILON or self.last_pwm > EPSILON:
                self.last_effective_pwm = value
                pin_updates.append((self.mcu_pwm_pin, value))
        elif abs(value - self.last_effective_pwm) >= self.PWM_WRITE_MIN_DELTA:
            if self._is_endpoint_reserved(print_time):
                self.mmu.log_stepper(
                    "BLDC_SET_PIN: discard pin=pwm value=%.4f print_time=%.6f reason=endpoint_reserve"
                    % (value, print_time)
                )
            else:
                self.last_effective_pwm = value
                pin_updates.append((self.mcu_pwm_pin, value))

        self._send_pin_updates(pin_updates, print_time)
        return len(pin_updates) > 1

    def _prepare_direction(self, start_time, direction):
        pin_value = 1 if direction > 0 else 0
        if self.last_dir == pin_value:
            return start_time
        self._safe_set_direction(pin_value, start_time)
        return max(start_time, self._last_dir_write_print_time) \
            + self.direction_setup_time

    def _get_planned_kick_time(self, move_duration=None):
        if self.kick_start_time <= EPSILON:
            return 0.
        queue_interval = self.mcu.min_schedule_time()
        kick_time = max(self.kick_start_time, queue_interval)
        if move_duration is not None \
                and move_duration < kick_time + queue_interval - EPSILON:
            return 0.
        return kick_time

    def queue_trapzoid_move(self, move, axis_r, print_time, source):
        if move.accel_t <= EPSILON and move.cruise_t <= EPSILON and move.decel_t <= EPSILON:
            return
        self._finite_move_tracker = None
        direction = 1 if move.cruise_v * axis_r > 0. else -1
        accel_mm_s2 = move.accel * abs(axis_r)
        start_v = move.start_v * axis_r
        cruise_v = move.cruise_v * axis_r
        end_v = move.end_v * axis_r
        if self._is_torque_assist_speed(cruise_v):
            mapped_forward = self._map_forward_for_gate(cruise_v > 0., cruise_v)
            assist_direction = 1 if mapped_forward else -1
            duration = move.accel_t + move.cruise_t + move.decel_t
            self.motion_queue.append((MotionTorqueAssist(
                print_time, self.torque_assist_pwm, duration, assist_direction,
            ), source))
            self.motion_state = self.MOTION_STATE_MOVING
            self._start_motion_timer()
            return
        if move.accel_t > EPSILON:
            self.motion_queue.append((MotionTrapzoid(
                print_time, start_v, cruise_v, accel_mm_s2, direction=direction,
            ), source))
        if move.cruise_t > EPSILON:
            self.motion_queue.append((MotionDescriptor(
                print_time + move.accel_t, cruise_v, True, direction, duration=move.cruise_t,
            ), source))
        if move.decel_t > EPSILON:
            self.motion_queue.append((MotionTrapzoid(
                print_time + move.accel_t + move.cruise_t, cruise_v, end_v, accel_mm_s2, direction=direction,
            ), source))
        self.motion_state = self.MOTION_STATE_MOVING
        self._start_motion_timer()

    def _get_scheduled_print_time(self):
        return self.mcu.estimated_print_time(self.reactor.monotonic())

    def _is_motion_active(self):
        return self.motion_state in (self.MOTION_STATE_MOVING, self.MOTION_STATE_BRAKE)

    def _calculate_brake_time(self, applied_pwm):
        if self.brake_pwm <= EPSILON or self.brake_max_time <= EPSILON:
            return 0.
        brake_scale = min(1., max(0., applied_pwm / max(self.pwm_max, EPSILON)))
        return max(self.BRAKE_MIN_TIME_S, brake_scale * self.brake_max_time)

    def _schedule_finite_move_endpoint(self, target_print_time):
        tracker = self._finite_move_tracker
        if tracker is None or tracker.endpoint_locked:
            return

        target_print_time = max(target_print_time, tracker.cruise_start_time)

        tracker.target_print_time = target_print_time
        tracker.cruise_descriptor.duration = max(
            0., target_print_time - tracker.cruise_descriptor.print_time
        )
        tracker.stop_descriptor.print_time = target_print_time
        if self._motion_timer_running and self.motion_timer is not None:
            self.reactor.update_timer(self.motion_timer, self.reactor.NOW)

    def _get_position_stop_lead_limit(self, tracker):
        revolution_limit = (
            self.tachometer.get_counts_per_revolution()
            * self.POSITION_MAX_STOP_LEAD_REVOLUTIONS
        )
        move_limit = (
            tracker.target_count_delta * self.POSITION_MAX_STOP_LEAD_RATIO
        )
        return min(revolution_limit, move_limit)

    def _finalize_position_learning(self, tracker=None):
        tracker = tracker or self._finite_move_tracker
        if tracker is None or tracker.aborted or not tracker.completed \
                or tracker.learning_completed or tracker.settle_timed_out \
                or tracker.settled_count is None:
            return

        travelled_counts = tracker.get_travelled_counts(tracker.settled_count)
        powered_duration = (
            tracker.actual_stop_print_time - tracker.start_print_time
        )
        if travelled_counts is None or travelled_counts <= 0. \
                or tracker.settled_sample_time <= tracker.start_print_time \
                or powered_duration <= 0.:
            return

        feed_forward = self._position_feed_forward[tracker.direction]
        observed_frequency = travelled_counts / powered_duration
        feed_forward.learn_frequency(
            observed_frequency,
            tracker.target_rpm,
        )
        position_error_counts = travelled_counts - tracker.target_count_delta
        feed_forward.learn_stop_lead(
            tracker.stop_lead_counts,
            position_error_counts,
            self._get_position_stop_lead_limit(tracker),
        )
        tracker.learning_completed = True
        self.mmu.log_stepper(
            "BLDC_POSITION_LEARN: travelled_counts=%.1f target_counts=%.1f "
            "error_counts=%.1f effective_freq=%.4f stop_lead_counts=%.2f "
            "commanded_rpm=%.1f direction=%d unit=%s"
            % (
                travelled_counts, tracker.target_count_delta,
                position_error_counts, feed_forward.effective_frequency,
                feed_forward.stop_lead_counts, feed_forward.commanded_rpm,
                tracker.direction, self.section_name,
            )
        )

    def handle_tachometer_position_sample(self, sample_time, count, count_time):
        tracker = self._finite_move_tracker
        if tracker is None:
            return
        if tracker.endpoint_locked:
            if tracker.settled_count is not None:
                return
            motor_stopped = self.tachometer.last_tach_rpm <= EPSILON
            if motor_stopped:
                settled = tracker.observe_standstill(count, sample_time)
            else:
                tracker.reset_standstill()
                settled = False
            self.mmu.log_stepper(
                "BLDC_POSITION_SETTLE: count=%.1f rpm=%.1f zero_samples=%d settled=%d print_time=%.6f unit=%s"
                % (
                    count, self.tachometer.last_tach_rpm,
                    tracker.standstill_sample_count, int(settled),
                    sample_time, self.section_name,
                )
            )
            if settled:
                self._finalize_position_learning(tracker)
            return
        if tracker.start_count is None:
            if sample_time <= tracker.start_print_time:
                tracker.set_start_count(count)
            return
        if sample_time < tracker.start_print_time:
            tracker.set_start_count(count)
            return

        travelled_counts = tracker.get_travelled_counts(count)
        if travelled_counts is None or travelled_counts <= 0.:
            return

        filtered_frequency = tracker.observe_progress(
            travelled_counts,
            count_time,
            self.POSITION_FREQUENCY_FILTER_WEIGHT,
        )
        if filtered_frequency is None:
            return

        if not tracker.has_frequency_seed:
            live_stop_lead_counts = (
                filtered_frequency
                * self.tachometer.tachometer_sample_time
            )
            tracker.stop_lead_counts = min(
                self._get_position_stop_lead_limit(tracker),
                max(
                    tracker.learned_stop_lead_counts,
                    live_stop_lead_counts,
                ),
            )

        tracker.no_progress_deadline = max(
            tracker.no_progress_deadline,
            count_time + self.POSITION_NO_PROGRESS_TIMEOUT_S,
        )

        remaining_counts = max(
            0., tracker.get_pwm_off_target_counts() - travelled_counts
        )
        target_print_time = (
            count_time + remaining_counts / filtered_frequency
        )
        target_print_time = min(target_print_time, tracker.no_progress_deadline)
        current_print_time = self._get_scheduled_print_time()
        predicted_current_counts = (
            travelled_counts
            + filtered_frequency * max(0., current_print_time - count_time)
        )
        earliest_schedulable_time = max(
            current_print_time + self.min_schedule_time,
            self._last_pwm_write_print_time + self.mcu.min_schedule_time(),
        )
        target_print_time = max(target_print_time, earliest_schedulable_time)
        self._schedule_finite_move_endpoint(target_print_time)
        self.mmu.log_stepper(
            "BLDC_POSITION: travelled_counts=%.1f predicted_counts=%.1f "
            "target_counts=%.1f stop_lead_counts=%.2f recent_freq=%.4f "
            "target_pt=%.6f unit=%s"
            % (
                travelled_counts, predicted_current_counts,
                tracker.target_count_delta, tracker.stop_lead_counts,
                filtered_frequency, tracker.target_print_time,
                self.section_name,
            )
        )

    def get_last_move_revolutions(self, requested_distance_mm):
        tracker = self._finite_move_tracker
        if tracker is None or tracker.aborted or not tracker.completed \
                or tracker.settled_count is None:
            return None
        if not tracker.matches_distance(
                requested_distance_mm, self.POSITION_DISTANCE_TOLERANCE_MM):
            return None
        self._finalize_position_learning(tracker)
        travelled_counts = tracker.get_travelled_counts(tracker.settled_count)
        if travelled_counts is None or travelled_counts <= 0.:
            return None
        return travelled_counts / self.tachometer.get_counts_per_revolution()

    def calculate_rotation_distance(self, measured_distance_mm, requested_distance_mm):
        measured_revolutions = self.get_last_move_revolutions(requested_distance_mm)
        if measured_revolutions is None:
            return None
        return measured_distance_mm / measured_revolutions

    def _handle_connect(self):
        """Probe and resolve sync monitor on connect (extruder guaranteed registered)."""
        if self.mmu is None:
            self.mmu = self.printer.lookup_object('mmu')
        self.tachometer.mmu = self.mmu
        owner = getattr(self, 'mmu_unit', None)
        owner = owner.extruder_wrapper if owner is not None else self.mmu
        try:
            monitor = getattr(owner, '_bldc_process_move_monitor', None)
            if monitor is None:
                monitor = ProcessMoveSyncMonitor(self.mmu)
                setattr(owner, '_bldc_process_move_monitor', monitor)
            if not monitor.activate(self):
                raise self.config.error(
                    "sync_monitor failed to activate in [%s]" % (self.section_name)
                )
            monitor.deactivate(self)
            self.active_sync_monitor = monitor
        except Exception as e:
            if 'error' in str(type(e).__name__).lower():
                raise
            raise self.config.error(
                "Failed to initialize sync monitor in [%s]: %s" % (self.section_name, str(e))
            )

    def _ensure_motion_timer(self):
        if self.motion_timer is None:
            self.motion_timer = self.reactor.register_timer(self._motion_timer_callback)

    def _start_motion_timer(self):
        """Start the motion timer only when it is currently stopped and motion is active."""
        if not self._motion_timer_running and self._is_motion_active():
            self._ensure_motion_timer()
            self._motion_timer_running = True
            self.reactor.update_timer(self.motion_timer, self.reactor.NOW)

    def _stop_motion_timer(self):
        if self.motion_timer is not None:
            self._motion_timer_running = False
            self.reactor.update_timer(self.motion_timer, self.reactor.NEVER)

    def _log_descriptor(self, desc, current_print_time):
        """Log the winning motion descriptor being dispatched.

        Logs: descriptor id, type, planned print_time, queued print_time, PWM/direction, creation location.
        """
        desc_id = id(desc)
        desc_type = type(desc).__name__
        planned_time = desc.print_time
        queued_time = current_print_time

        # Extract PWM and direction based on descriptor type
        if isinstance(desc, MotionStop):
            pwm_val, dir_val = 0.0, 0
        elif isinstance(desc, MotionPwmDirect):
            pwm_val, dir_val = desc.pwm, (1 if desc.direction > 0 else 0)
        elif isinstance(desc, MotionTrapzoid):
            speed = desc.get_speed(current_print_time)
            pwm_val = None  # Speed-mode; PWM computed from RPM later
            dir_val = 1 if speed > 0 else (0 if speed < 0 else (self.last_dir or 0))
        else:
            pwm_val, dir_val = None, None

        # Extract creation location from traceback (line number and source code)
        creation_loc = "unknown"
        # Get the frame before descriptor creation (typically where append was called)
        for frame_str in reversed(desc.trackback[:-2]):  # Skip last 2 (module initialization)
            if 'mmu_gear_bldc' in frame_str:
                # Extract line number and code from traceback frame
                lines = frame_str.strip().split('\n')
                if len(lines) >= 2:
                    location_line = lines[0]  # "  File "...", line 123, in func"
                    code_line = lines[1].strip()  # Actual source code
                    # Extract line number from "line 123"
                    if 'line ' in location_line:
                        line_num = location_line.split('line ')[1].split(',')[0]
                        creation_loc = f"L{line_num}:{code_line}"
                break

        msg = (f"BLDC_DESC: id={desc_id:08x} type={desc_type:<16} "
               f"planned_pt={planned_time:.6f} queued_pt={queued_time:.6f} src={creation_loc}")
        if pwm_val is not None:
            msg += f" pwm={pwm_val:.3f}"
        if dir_val is not None:
            msg += f" dir={dir_val}"

        self.mmu.log_stepper(msg)

    def _get_motion_timer_wake(self, eventtime, current_print_time):
        future_descriptors = [
            descriptor.print_time
            for descriptor, _source in self.motion_queue
            if descriptor.print_time > current_print_time + EPSILON
        ]
        if not future_descriptors:
            return eventtime + self.motion_sample_time
        time_to_horizon = min(future_descriptors) - current_print_time
        return eventtime + min(self.motion_sample_time, time_to_horizon)

    def _stop_synced_drive(self, source, print_time=None):
        self._queue_pwm_if_changed(0., print_time)
        self._reset_motion_command(source)

    def _motion_timer_callback(self, eventtime):
        if not self._is_motion_active():
            self._motion_timer_running = False
            return self.reactor.NEVER

        # Base dispatch on the live clock, not the reactor-provided eventtime, so writes stay ahead
        # of the real MCU clock even after the gear+extruder quiesce stalls the reactor.
        current_print_time = self._get_scheduled_print_time() + self.min_schedule_time

        # Prune descriptors whose full time window has elapsed
        kept_motion_queue = []
        for d, src in self.motion_queue:
            expire_time = d.print_time + d.duration
            if expire_time > current_print_time + EPSILON:
                kept_motion_queue.append((d, src))
                continue
            self.mmu.log_stepper(
                "BLDC_DESC_PRUNE: id=%08x type=%s src=%s planned_pt=%.6f expire_pt=%.6f now_pt=%.6f"
                % (id(d), type(d).__name__, src, d.print_time, expire_time, current_print_time)
            )
        self.motion_queue = kept_motion_queue

        if not self.motion_queue:
            self._motion_timer_running = False
            # In process_move sync mode (GEAR_SYNCED_TO_EXTRUDER), the queue draining means the
            # extruder has stopped moving -- stop the BLDC immediately. In event-driven mode
            # (dist=None, sync_active=False) the caller holds the queue open with an INFINITY
            # descriptor, so this branch is never reached while a move is in progress.
            if self.sync_active:
                self._stop_synced_drive('sync_queue_drained')
            return self.reactor.NEVER

        # Filter to the next-tick window (early send: ensure all candidates within MCU schedule constraint)
        candidates = [
            (d, src) for d, src in self.motion_queue
            if d.print_time <= current_print_time
        ]

        if not candidates:
            if self.sync_active and self.last_effective_pwm > EPSILON:
                self._stop_synced_drive('sync_gap', current_print_time)
            return self._get_motion_timer_wake(eventtime, current_print_time)

        # Winner selection: drop MotionStop if any non-stop present
        non_stop = [(d, src) for d, src in candidates if not isinstance(d, MotionStop)]
        if non_stop:
            candidates = non_stop

        # Filter out colliding descriptors (those < 0.001s from previous); batch all remaining
        sorted_candidates = sorted(candidates, key=lambda x: x[0].print_time)
        batch = []
        for d, src in sorted_candidates:
            if not batch or (d.print_time - batch[-1][0].print_time >= 0.001):
                batch.append((d, src))

        # Dispatch all non-colliding descriptors in batch
        for idx, (desc, src) in enumerate(batch):
            is_last_in_batch = idx == len(batch) - 1
            dispatch_time = desc.print_time if desc.print_time >= current_print_time else current_print_time
            if isinstance(desc, MotionStop):
                if is_last_in_batch:
                    self._motion_timer_running = False
                tracker = self._finite_move_tracker
                if tracker is not None and desc is tracker.stop_descriptor:
                    tracker.endpoint_locked = True
                    tracker.stop_dispatched = True
                    tracker.completed = True
                    tracker.actual_stop_print_time = dispatch_time
                if self._queue_pwm_if_changed(0., dispatch_time):
                    self._log_descriptor(desc, dispatch_time)
                self.motion_state = self.MOTION_STATE_STOP
                self.motion_queue = []
                self._reset_motion_command('stop')
            elif desc.pwm is not None:
                if isinstance(desc, MotionTorqueAssist):
                    self._activate_torque_assist()
                elif src == 'brake':
                    self.motion_state = self.MOTION_STATE_BRAKE
                    self._reset_motion_command(src)
                dir_val = 1 if desc.direction > 0 else 0
                if self._queue_drive_if_changed(
                        dir_val, desc.pwm, dispatch_time):
                    self._log_descriptor(desc, dispatch_time)
            else:
                speed_mm_s = desc.get_speed(current_print_time) if isinstance(desc, MotionTrapzoid) else desc.speed_mm_s
                if abs(speed_mm_s) < EPSILON:
                    queued_pwm = self._queue_pwm_if_changed(0., dispatch_time)
                    self._reset_motion_command(src)
                    if queued_pwm:
                        self._log_descriptor(desc, dispatch_time)
                else:
                    forward = self._map_forward_for_gate(speed_mm_s > 0., speed_mm_s)
                    requested_rpm = 60. * abs(speed_mm_s) / self.rotation_distance
                    rpm = max(0., min(requested_rpm, self.get_effective_max_rpm()))
                    if rpm > EPSILON and self.commanded_rpm > EPSILON \
                            and abs(rpm - self.commanded_rpm) > self.LARGE_SPEED_CHANGE_RPM:
                        self.tachometer.reset_integral()
                        self._last_pwm_write_print_time = 0.
                    elif rpm > EPSILON >= self.commanded_rpm:
                        self._last_pwm_write_print_time = 0.
                    self.commanded_rpm = rpm
                    self.commanded_source = src
                    self.commanded_linear_speed = abs(speed_mm_s)
                    self.tachometer.set_commanded(rpm, src)
                    self.tachometer.enabled = desc.pid_enable
                    pwm = self.rpm_to_pwm(rpm)
                    effective_pwm = self.tachometer.apply_control(pwm)
                    if self._queue_drive_if_changed(
                            int(forward), effective_pwm, dispatch_time):
                        self._log_descriptor(desc, dispatch_time)

        # Check if motion should stop (MotionStop was in batch and is last)
        if batch and isinstance(batch[-1][0], MotionStop):
            return self.reactor.NEVER

        return self._get_motion_timer_wake(eventtime, current_print_time)

    def _reset_motion_command(self, source):
        self.commanded_rpm = 0.
        self.commanded_source = source
        self.commanded_linear_speed = 0.
        self.last_effective_pwm = 0.
        self._last_pwm_write_print_time = 0.
        self.tachometer.stop()
        self.tachometer.enabled = False

    def _activate_torque_assist(self):
        self.commanded_rpm = 0.
        self.commanded_linear_speed = 0.
        self.commanded_source = 'torque_assist'
        self.tachometer.stop()
        self.tachometer.enabled = False

    def _reset_motion(self, state, source):
        self.motion_state = state
        self._stop_motion_timer()
        self.motion_queue = []
        self._reset_motion_command(source)

    def stop(self, print_time=None):
        tracker = self._finite_move_tracker
        if tracker is not None and not tracker.completed:
            tracker.aborted = True
        self._reset_motion(self.MOTION_STATE_STOP, 'stop')
        if abs(self.last_pwm) > EPSILON:
            self._send_pin(self.mcu_pwm_pin, 0., print_time)
        toolhead = self.printer.lookup_object('toolhead', None)
        if toolhead is not None and hasattr(toolhead, 'flush_step_generation'):
            toolhead.flush_step_generation()

    def brake_to_stop(self):
        self._finite_move_tracker = None
        print_time = self._get_scheduled_print_time() + self.min_schedule_time
        applied_pwm = abs(self.last_pwm)
        if applied_pwm < self.BRAKE_MIN_ACTIVE_PWM or self.last_dir is None:
            self.stop(print_time)
            return

        brake_time = self._calculate_brake_time(applied_pwm)
        if brake_time <= EPSILON:
            self.stop(print_time)
            return

        brake_direction = -1 if self.last_dir else 1
        self._reset_motion(self.MOTION_STATE_BRAKE, 'brake')
        brake_descriptor = MotionPwmDirect(
            print_time, self.brake_pwm, brake_time, brake_direction
        )
        stop_descriptor = MotionStop(print_time + brake_time)
        self.motion_queue.append((brake_descriptor, 'brake'))
        self.motion_queue.append((stop_descriptor, 'brake'))
        self._start_motion_timer()

    def _queue_open_ended_move(self, requested_speed, start_time):
        requested_forward = requested_speed >= 0.
        direction = 1 if self._map_forward_for_gate(
            requested_forward, requested_speed
        ) else -1
        start_time = self._prepare_direction(start_time, direction)
        if self._is_torque_assist_speed(requested_speed):
            self.motion_queue.append((MotionTorqueAssist(
                start_time, self.torque_assist_pwm, INFINITY, direction,
            ), 'move'))
            return
        logical_speed = requested_speed
        kick_time = self._get_planned_kick_time()
        if kick_time > EPSILON:
            self.motion_queue.append((
                MotionPwmDirect(start_time, self.pwm_max, kick_time, direction),
                'move',
            ))
        cruise_start_time = start_time + kick_time
        self.motion_queue.append((
            MotionDescriptor(
                cruise_start_time, logical_speed, True, direction,
                duration=INFINITY,
            ),
            'move',
        ))

    def _get_finite_move_speed(self, requested_speed):
        effective_speed = self._get_achievable_linear_speed(requested_speed)
        calibrated_min_rpm = self._get_calibrated_min_rpm() or 0.
        if effective_speed < requested_speed - EPSILON:
            self.mmu.log_warning(
                "BLDC gear speed limited from %.3fmm/s to %.3fmm/s by %.1f RPM maximum (%s)"
                % (
                    requested_speed, effective_speed,
                    self.get_effective_max_rpm(), self.section_name,
                )
            )
        elif calibrated_min_rpm > EPSILON and effective_speed > requested_speed + EPSILON:
            self.mmu.log_warning(
                "BLDC gear speed increased from %.3fmm/s to %.3fmm/s by %.1f RPM minimum (%s)"
                % (
                    requested_speed, effective_speed,
                    calibrated_min_rpm, self.section_name,
                )
            )
        return effective_speed

    def _queue_position_guided_move(self, tracker, logical_speed, direction):
        cruise_duration = tracker.target_print_time - tracker.cruise_start_time
        cruise_descriptor = MotionDescriptor(
            tracker.cruise_start_time, logical_speed, True, direction,
            duration=cruise_duration,
        )
        tracker.cruise_descriptor = cruise_descriptor
        self.motion_queue.append((cruise_descriptor, 'move'))

        tracker.stop_descriptor = MotionStop(tracker.target_print_time)
        self.motion_queue.append((tracker.stop_descriptor, 'move'))
        self._finite_move_tracker = tracker
        self._schedule_finite_move_endpoint(tracker.target_print_time)

    def _queue_finite_move(self, dist, requested_speed, start_time):
        mapped_dist = self._map_distance_for_gate(dist)[0]
        direction = 1 if mapped_dist > 0. else -1
        start_time = self._prepare_direction(start_time, direction)
        effective_speed = self._get_finite_move_speed(requested_speed)
        logical_speed = effective_speed if dist > 0. else -effective_speed
        move_duration = abs(mapped_dist) / effective_speed

        kick_time = self._get_planned_kick_time(move_duration)
        if kick_time > EPSILON:
            self.motion_queue.append((
                MotionPwmDirect(start_time, self.pwm_max, kick_time, direction),
                'move',
            ))

        cruise_start_time = start_time + kick_time
        cruise_duration = move_duration - kick_time
        start_count, _count_time = self.tachometer.get_count_sample()
        if start_count is None:
            if cruise_duration > EPSILON:
                self.motion_queue.append((
                    MotionDescriptor(
                        cruise_start_time, logical_speed, True, direction,
                        duration=cruise_duration,
                    ),
                    'move',
                ))
            stop_descriptor = MotionStop(start_time + move_duration)
            self.motion_queue.append((stop_descriptor, 'move'))
            return stop_descriptor.print_time - self._get_scheduled_print_time()

        target_count_delta = (
            abs(mapped_dist)
            * self.tachometer.get_counts_per_revolution()
            / self.rotation_distance
        )
        tracker = FiniteMoveTracker(mapped_dist, start_time, target_count_delta)
        tracker.set_start_count(start_count)
        tracker.direction = direction
        tracker.target_rpm = 60. * effective_speed / self.rotation_distance
        feed_forward = self._position_feed_forward[direction]
        tracker.stop_lead_counts = min(
            feed_forward.stop_lead_counts,
            self._get_position_stop_lead_limit(tracker),
        )
        tracker.learned_stop_lead_counts = tracker.stop_lead_counts
        tracker.filtered_frequency = feed_forward.estimate_frequency(
            tracker.target_rpm
        )
        tracker.has_frequency_seed = tracker.filtered_frequency > EPSILON
        tracker.cruise_start_time = cruise_start_time
        nominal_target_print_time = start_time + move_duration
        tracker.nominal_target_print_time = nominal_target_print_time
        tracker.target_print_time = nominal_target_print_time
        if tracker.filtered_frequency > EPSILON:
            tracker.target_print_time = (
                start_time
                + tracker.target_count_delta
                / tracker.filtered_frequency
            )
        tracker.no_progress_deadline = (
            tracker.target_print_time + self.POSITION_NO_PROGRESS_TIMEOUT_S
        )
        self.mmu.log_stepper(
            "BLDC_POSITION_SEED: target_counts=%.1f stop_lead_counts=%.2f "
            "feed_forward_freq=%.4f nominal_pt=%.6f target_pt=%.6f "
            "direction=%d unit=%s"
            % (
                tracker.target_count_delta, tracker.stop_lead_counts,
                tracker.filtered_frequency, nominal_target_print_time,
                tracker.target_print_time, tracker.direction,
                self.section_name,
            )
        )
        self._queue_position_guided_move(tracker, logical_speed, direction)
        return (
            tracker.stop_descriptor.print_time
            - self._get_scheduled_print_time()
        )

    def start_move(self, dist, speed, print_time=None):
        """Queue motion and return the required host wait for a finite move."""
        self._finalize_position_learning()
        requested_speed = speed
        speed = abs(speed)
        if (dist is not None and dist == 0.) or speed <= EPSILON:
            self.stop()
            return 0.
        self._finite_move_tracker = None
        self.motion_queue = []
        self._reset_motion_command('move_start')
        start_time = self._get_scheduled_print_time() + self.min_schedule_time
        if print_time is not None:
            start_time = max(start_time, print_time)

        if dist is None:
            self._queue_open_ended_move(requested_speed, start_time)
            move_wait = None
        else:
            move_wait = self._queue_finite_move(dist, speed, start_time)
        self.motion_state = self.MOTION_STATE_MOVING
        self._start_motion_timer()
        return move_wait

    def wait_for_move(self, fallback_wait):
        tracker = self._finite_move_tracker
        if tracker is None:
            wait_time = fallback_wait + self.motion_sample_time
            self.reactor.pause(self.reactor.monotonic() + wait_time)
            return

        poll_interval = min(
            self.POSITION_WAIT_POLL_MAX_S,
            max(self.motion_sample_time, self.tachometer.tachometer_sample_time),
        )
        while True:
            current_print_time = self._get_scheduled_print_time()
            if tracker.stop_dispatched:
                if tracker.settled_count is not None:
                    self._finalize_position_learning(tracker)
                    return
                stop_print_time = tracker.actual_stop_print_time \
                    or tracker.stop_descriptor.print_time
                settle_deadline = (
                    stop_print_time + self.POSITION_STANDSTILL_TIMEOUT_S
                )
                if current_print_time >= settle_deadline:
                    tracker.settle_timed_out = True
                    self.mmu.log_warning(
                        "BLDC position move did not reach two zero-RPM samples after stop (%s)"
                        % self.section_name
                    )
                    return
                remaining_time = settle_deadline - current_print_time
                self.reactor.pause(
                    self.reactor.monotonic()
                    + min(poll_interval, remaining_time)
                )
                continue
            if current_print_time >= tracker.no_progress_deadline:
                self.mmu.log_warning(
                    "BLDC position move made no tach progress before stop deadline (%s)"
                    % self.section_name
                )
                self.stop()
                self.reactor.pause(
                    self.reactor.monotonic() + self.min_schedule_time
                )
                return
            self.reactor.pause(self.reactor.monotonic() + poll_interval)

    def set_rotation_distance(self, value):
        if value > 0.:
            self.rotation_distance = value

    def has_tachometer(self):
        return self.tachometer.has_tachometer()

    def get_rotation_distance(self):
        return self.rotation_distance

    def get_status(self, _eventtime):
        tach_status = self.tachometer.get_status()
        return {
            'active': abs(self.last_pwm) > EPSILON, 'pwm': self.last_pwm, 'dir': self.last_dir,
            'torque_assist_active': self.commanded_source == 'torque_assist',
            'torque_assist_pwm': self.torque_assist_pwm,
            'sync_retract_min_distance': self.sync_retract_min_distance,
            'rotation_distance': self.rotation_distance,
            'tachometer_frequency': self.tachometer.last_tach_frequency, 'tachometer_rpm': tach_status['tachometer_rpm'],
            'tachometer_fresh': self.tachometer.has_fresh_tachometer(), 'commanded_rpm': self.commanded_rpm,
            'tachometer_error_rpm': tach_status['tachometer_error_rpm'],
            'control_enabled': tach_status['control_enabled'], 'control_reason': tach_status['control_reason'],
            'control_correction_pwm': tach_status['control_correction_pwm'],
            'integral_correction_pwm': tach_status['integral_correction_pwm'], 'effective_pwm': self.last_effective_pwm,
            'map_mode': self.map_mode, 'map_points': len(self.calibration_map_points), 'map_fallback_reason': self.map_fallback_reason,
            'effective_max_rpm': self.get_effective_max_rpm(),
            'calibrated_max_rpm': self._get_calibrated_max_rpm(),
            'motion_sample_time': self.motion_sample_time,
        }

    def _handle_synced(self):
        self.sync_active = True
        if self.active_sync_monitor is not None:
            self.active_sync_monitor.activate(self)

    def _handle_unsynced(self):
        self.sync_active = False
        if self.active_sync_monitor is not None:
            self.active_sync_monitor.deactivate(self)
        if self.motion_state == self.MOTION_STATE_MOVING and self.motion_queue:
            return
        self.stop()

    def _handle_shutdown(self):
        if self.active_sync_monitor is not None:
            self.active_sync_monitor.deactivate(self)
        self.stop()

    def set_sync_enabled(self, enabled):
        (self._handle_synced if enabled else self._handle_unsynced)()
