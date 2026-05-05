# Happy Hare MMU Software
#
# BLDC gear controller for MMU gear motion replacement.
#
# Copyright (C) 2022-2026  moggieuk#6538 (discord)
#                          moggieuk@hotmail.com
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#

from types import MethodType
import inspect
from .mmu_shared import MmuError
from .. import pulse_counter
from ..pwm_tool import MCU_queued_pwm

EPSILON = 1e-6
INFINITY = float('inf')

class MotionDescriptor:
    """Base motion descriptor with direction and speed mode."""
    def __init__(self, speed_mm_s, pid_enable, print_time, direction=1):
        self.speed_mm_s = speed_mm_s
        self.pid_enable = pid_enable
        self.print_time = print_time
        self.direction = direction  # 1=forward, -1=backward
        self.pwm = None  # Only MotionPwmDirect sets this; indicates PWM-direct dispatch mode

    def advance(self, print_time):
        """Called after dispatch. Update state and return wake delay (seconds) or None."""
        return None

class MotionStop(MotionDescriptor):
    """Terminal stop descriptor (speed=0, pid=False, direction-agnostic)."""
    def __init__(self, print_time):
        super().__init__(0., False, print_time)

    def advance(self, print_time):
        """Return INFINITY to signal queue termination and timer stop."""
        return INFINITY

class MotionKickToSpeed(MotionDescriptor):
    """Two-phase descriptor: PWM kick → speed control."""
    def __init__(self, direction, kick_start_time, target_speed_mm_s, print_time):
        super().__init__(0., False, print_time, direction=direction)
        self.transition_print_time = print_time + max(0., kick_start_time)
        self.target_speed_mm_s = target_speed_mm_s
        self.pwm = 1.0  # Start in PWM-direct mode

    def advance(self, print_time):
        """Manage kick→speed phase transition."""
        if self.pwm is None:
            return None
        if print_time >= self.transition_print_time - EPSILON:
            self.pwm = None
            self.speed_mm_s = self.target_speed_mm_s
        else:
            return self.transition_print_time - print_time
        return None

class MotionPwmDirect(MotionDescriptor):
    """PWM-direct mode descriptor (for brake and other direct PWM control)."""
    def __init__(self, direction, pwm, duration, print_time):
        super().__init__(0., False, print_time, direction=direction)
        self.pwm = pwm  # Set PWM directly (base class sets pwm=None)
        self.duration = duration  # How long to apply this PWM

    def advance(self, print_time):
        """Return wake delay so timer fires when this PWM phase expires."""
        return self.duration if self.duration else None

class MotionTrapzoid(MotionDescriptor):
    """Trapezoidal accel/decel profile (for process_move sync)."""
    def __init__(self, direction, start_speed_mm_s, end_speed_mm_s, accel_mm_s2, *, print_time):
        super().__init__(start_speed_mm_s, True, print_time, direction=direction)
        self.start_speed_mm_s = start_speed_mm_s
        self.end_speed_mm_s = end_speed_mm_s
        self.accel_mm_s2 = accel_mm_s2
        self.end_print_time = (print_time + abs(end_speed_mm_s - start_speed_mm_s) / accel_mm_s2
                               if accel_mm_s2 > EPSILON else print_time)

    def _get_speed_at_time(self, print_time):
        """Compute speed at given print_time within profile."""
        if print_time <= self.print_time:
            return self.start_speed_mm_s
        if print_time >= self.end_print_time:
            return self.end_speed_mm_s
        else:
            accel_time = print_time - self.print_time
            accel_speed = accel_time * self.accel_mm_s2
            if self.end_speed_mm_s > self.start_speed_mm_s:
                return self.start_speed_mm_s + accel_speed
            else:
                return self.start_speed_mm_s - accel_speed

    def advance(self, print_time):
        """Update speed within profile; poll at 100ms intervals or phase end."""
        self.speed_mm_s = self._get_speed_at_time(print_time)
        if print_time >= self.end_print_time:
            self.print_time = None
            return None
        return min(0.1, self.end_print_time - print_time)

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
        if self.hook_enabled and self.active_bldc is bldc:
            return True

        extruder = self.mmu.toolhead.get_extruder()
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

        def wrapped_process_move(hooked_self, print_time, move, ea_index):
            process_move(print_time, move, ea_index)
            self._handle_process_move(print_time, move, ea_index)

        extruder._hh_bldc_original_process_move = process_move
        extruder._hh_bldc_process_move_owner = self
        extruder.process_move = MethodType(wrapped_process_move, extruder)
        self.hooked_extruder = extruder
        self.original_process_move = process_move
        self._attach_bldc(bldc)
        self.mmu.log_stepper("BLDC_PROCESS_MOVE: hook installed")
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
            self.mmu.log_stepper("BLDC_PROCESS_MOVE: hook removed")

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

        start_v = move.start_v * axis_r
        cruise_v = move.cruise_v * axis_r
        end_v = move.end_v * axis_r
        accel = move.accel if hasattr(move, 'accel') else 0.

        self.mmu.log_stepper("BLDC_PROCESS_MOVE: time=%.3f start_v=%.3f cruise_v=%.3f end_v=%.3f accel=%.1f accel_t=%.4f cruise_t=%.4f decel_t=%.4f unit=%s" % (print_time, start_v, cruise_v, end_v, accel, move.accel_t, move.cruise_t, move.decel_t, bldc.section_name))
        bldc.queue_trapzoid_move(move, axis_r, print_time, 'process_move_push')


class BldcTachometer:
    """Own tachometer sampling and PID PWM correction for BLDC."""

    TACH_VALID_MAX_AGE = 0.5
    CONTROL_DEADBAND_RPM = 50.
    CONTROL_MIN_RPM = 150.
    LOG_INTERVAL = 0.5

    PID_PARAM_BASE = 255.

    def __init__(self, bldc, config):
        self.mmu = bldc.mmu
        self.section_name = bldc.section_name
        self.reactor = bldc.reactor
        self.mcu_pwm_pin = bldc.mcu_pwm_pin
        self.bldc = bldc

        self.pwm_min = config.getfloat('pwm_min', 0.85, minval=0., maxval=1.)
        self.pwm_max = config.getfloat('pwm_max', 1.0, minval=0., maxval=1.)
        self.tachometer_ppr = config.getint('tachometer_ppr', 9, minval=1)
        self.tachometer_sample_time = config.getfloat('tachometer_sample_time', 0.1, above=0.)
        self.tachometer_stale_time = config.getfloat('tachometer_stale_time', self.tachometer_sample_time * 3., above=0.)
        self.control_kp = config.getfloat('tachometer_control_kp', 20.0, minval=0.) / self.PID_PARAM_BASE
        self.control_ki = config.getfloat('tachometer_control_ki', 1125.0, minval=0.) / self.PID_PARAM_BASE
        self.control_max_delta_pwm = config.getfloat('tachometer_control_max_delta_pwm', 0.15, minval=0., maxval=1.)

        tachometer_pin = config.get('tachometer_pin', None)
        self.tachometer = None
        if tachometer_pin is not None:
            self.tachometer = pulse_counter.MCU_counter(
                bldc.printer, tachometer_pin, self.tachometer_sample_time,
                config.getfloat('tachometer_poll_interval', 0.001, above=0.0009)
            )

        self.commanded_rpm = 0.
        self.commanded_source = None
        self.last_tach_freq = self.last_tach_rpm = self.last_tach_error_rpm = 0.
        self.last_tach_eventtime = None
        self.last_tach_log_eventtime = self.last_control_log_eventtime = None
        self.last_control_log_reason = None
        self.control_correction_pwm = self.integral_correction_pwm = 0.
        self.control_reason = 'disabled'
        self._tach_last_count = None
        self._tach_last_count_time = None
        self.enabled = False

        if self.tachometer is not None:
            self.tachometer.setup_callback(self.handle_tachometer)

    def has_tachometer(self):
        return self.tachometer is not None

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
        self.log_control('stop', 0., 0.)

    def reset_integral(self):
        self.integral_correction_pwm = 0.
        self.control_correction_pwm = 0.

    def apply_control(self, pwm, source):
        if pwm <= EPSILON:
            self.set_control_state('zero_pwm', 0., 0.)
            self.log_control(source, 0., 0.)
            return 0.

        reason = self._get_pid_skip_reason()
        if reason is not None:
            self.set_control_state(reason, 0.)
            self.log_control(source, pwm, pwm)
            return pwm

        applied_pwm = min(self.pwm_max, max(self.pwm_min, pwm + self.control_correction_pwm))
        self.set_control_state('active', self.control_correction_pwm, self.last_tach_error_rpm)
        self.log_control(source, pwm, applied_pwm)
        return applied_pwm

    def get_current_print_time(self):
        return self.mcu_pwm_pin.get_mcu().estimated_print_time(self.reactor.monotonic())

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
            'tachometer_frequency': self.last_tach_freq,
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

    def log_control(self, source, base_pwm, applied_pwm):
        eventtime = self.reactor.monotonic()
        reason_changed = self.control_reason != self.last_control_log_reason
        time_elapsed = self.last_control_log_eventtime is None \
            or eventtime - self.last_control_log_eventtime >= self.LOG_INTERVAL
        if not reason_changed and (not time_elapsed or self.control_reason in ('pid_disabled', 'stopped')):
            return
        self.last_control_log_eventtime = eventtime
        self.last_control_log_reason = self.control_reason
        print_time = self.mcu_pwm_pin.get_mcu().estimated_print_time(eventtime)
        self.mmu.log_stepper("BLDC_CONTROL: source=%s reason=%s error_rpm=%.1f base_pwm=%.4f correction_pwm=%.4f integral_pwm=%.4f applied_pwm=%.4f time=%.3f unit=%s" % (source, self.control_reason, self.last_tach_error_rpm, base_pwm, self.control_correction_pwm, self.integral_correction_pwm, applied_pwm, print_time, self.section_name))

    def handle_tachometer(self, time, count, count_time):
        if self._tach_last_count is None:
            self._tach_last_count = count
            self._tach_last_count_time = count_time
            self.last_tach_eventtime = time
            return
        delta_time = count_time - self._tach_last_count_time
        frequency = (count - self._tach_last_count) / delta_time if delta_time > 0. else 0.
        control_dt = delta_time if delta_time > 0. else (max(0., time - self.last_tach_eventtime) if self.last_tach_eventtime is not None else 0.)
        if control_dt <= 0.:
            control_dt = self.tachometer_sample_time
        self._tach_last_count = count
        self._tach_last_count_time = count_time
        self.last_tach_eventtime = time
        self.last_tach_freq = frequency
        self.last_tach_rpm = frequency * 30. / self.tachometer_ppr
        error_rpm = self.commanded_rpm - self.last_tach_rpm

        reason = self._get_pid_skip_reason()
        if reason is not None:
            self.integral_correction_pwm = 0.
            self.set_control_state(reason, 0., error_rpm)
            self.log_control(reason, 0., 0.)
        else:
            base_pwm = self.bldc.rpm_to_pwm(self.commanded_rpm)
            if abs(error_rpm) <= self.CONTROL_DEADBAND_RPM:
                self.set_control_state('deadband', self.control_correction_pwm, error_rpm)
                self.log_control('deadband', base_pwm, min(self.pwm_max, max(self.pwm_min, base_pwm + self.control_correction_pwm)))
            else:
                c = self.control_max_delta_pwm
                norm = (error_rpm / self.bldc.get_effective_max_rpm()) * (self.pwm_max - self.pwm_min)
                p_correction = max(-c, min(c, self.control_kp * norm))
                self.integral_correction_pwm = max(-c, min(c, self.integral_correction_pwm + self.control_ki * norm * control_dt))
                total_correction = max(-c, min(c, p_correction + self.integral_correction_pwm))
                self.set_control_state('active', total_correction, error_rpm)
                new_effective_pwm = min(self.pwm_max, max(self.pwm_min, base_pwm + total_correction))
                self.log_control('active', base_pwm, new_effective_pwm)
        self.log_tachometer()

    def log_tachometer(self):
        eventtime = self.reactor.monotonic()
        if self.last_tach_log_eventtime is not None \
                and eventtime - self.last_tach_log_eventtime < self.LOG_INTERVAL:
            return
        self.last_tach_log_eventtime = eventtime
        if self.commanded_rpm < 100 and self.last_tach_rpm < 100:
            return
        tach_print_time = self.last_tach_eventtime or self.mcu_pwm_pin.get_mcu().estimated_print_time(eventtime)
        self.mmu.log_stepper("BLDC_TACH: freq=%.3f rpm=%.1f time=%.3f unit=%s" % (self.last_tach_freq, self.last_tach_rpm, tach_print_time, self.section_name))


class MmuGearBldc:
    """BLDC gear controller for MMU gear motion replacement. Designed to be used with a BLDC motor and external ESC for gear drive sections, with optional tachometer feedback and closed loop control."""

    CALIBRATION_DEFAULT_POINTS = 16
    CALIBRATION_DEFAULT_SAMPLE_S = 1.0
    CALIBRATION_DEFAULT_SETTLE_S = 0.35
    CALIBRATION_MIN_POINTS = 3
    UNCALIBRATED_MAX_RPM = 5500.
    LARGE_SPEED_CHANGE_RPM = 500.
    PWM_WRITE_MIN_INTERVAL_S = 0.1
    PWM_WRITE_MIN_DELTA = 0.002
    BRAKE_MIN_TIME_S = 0.03
    BRAKE_MIN_ACTIVE_PWM = 0.08
    LOG_INTERVAL = 0.5

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
        self.last_sync_speed_log_eventtime = self.last_sync_sample_log_eventtime = None
        self.motion_state = self.MOTION_STATE_IDLE
        self.motion_queue = []
        self.motion_timer = None

        self.last_pwm = self.last_effective_pwm = 0.
        self._last_pwm_write_print_time = 0.
        self.last_dir = None
        self.section_name = config.get_name()
        self.commanded_rpm = self.commanded_linear_speed = 0.
        self.commanded_source = None
        self.calibration_map_points = []
        self.map_mode = 'linear'
        self.map_fallback_reason = 'map_missing'
        self.last_map_log_state = None

        self.kick_start_time = config.getfloat('kick_start_time', 0.05, minval=0.)
        self.brake_pwm = config.getfloat('brake_pwm', 1., minval=0., maxval=1.)
        self.brake_max_time = config.getfloat('brake_max_time', 0.25, minval=0.)

        self.pwm_min = config.getfloat('pwm_min', 0.85, minval=0., maxval=1.)
        self.pwm_max = config.getfloat('pwm_max', 1.0, minval=0., maxval=1.)
        if self.pwm_min > self.pwm_max:
            raise config.error("'pwm_min' cannot be greater than 'pwm_max' in [%s]" % config.get_name())

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

        self.tachometer = BldcTachometer(self, config)
        self.motion_sample_time = max(
            self.mcu_pwm_pin.get_mcu().min_schedule_time(),
            self.mcu_dir_pin.get_mcu().min_schedule_time(),
        )

        self.active_sync_monitor = None
        self.printer.register_event_handler('klippy:connect', self._handle_connect)
        self.printer.register_event_handler('mmu:synced', self._handle_synced)
        self.printer.register_event_handler('mmu:unsynced', self._handle_unsynced)
        self.printer.register_event_handler('klippy:shutdown', self._handle_shutdown)

    def _load_direction_map(self, config):
        values = list(config.getintlist('direction_map', []))
        if not values:
            return [0] * self.num_gates
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

    def _log_sync_sample(self, source, dt, de, raw_speed, selected_speed, speed_scale, moving):
        eventtime = self.reactor.monotonic()
        if self.last_sync_sample_log_eventtime is not None \
                and eventtime - self.last_sync_sample_log_eventtime < self.LOG_INTERVAL:
            return
        self.last_sync_sample_log_eventtime = eventtime
        self.mmu.log_stepper("BLDC_SYNC_SAMPLE: source=%s dt=%.4f de=%.4f raw_speed=%.3f selected_speed=%.3f scale=%.3f moving=%d unit=%s" % (source, dt, de, raw_speed, selected_speed, speed_scale, 1 if moving else 0, self.section_name))

    def _log_map_state(self, mode, reason):
        state = (mode, reason, len(self.calibration_map_points))
        if state == self.last_map_log_state:
            return
        self.last_map_log_state = state
        self.mmu.log_stepper("BLDC_MAP: mode=%s reason=%s points=%d unit=%s" % (mode, reason, len(self.calibration_map_points), self.section_name))

    def _log_speed(self, source, speed, requested_rpm, clamped_rpm, pwm, forward):
        eventtime = self.reactor.monotonic()
        if source == 'sync':
            if self.last_sync_speed_log_eventtime is not None \
                    and eventtime - self.last_sync_speed_log_eventtime < self.LOG_INTERVAL:
                return
            self.last_sync_speed_log_eventtime = eventtime
        self.mmu.log_stepper("BLDC_SPEED: source=%s speed=%.3f requested_rpm=%.1f clamped_rpm=%.1f pwm=%.4f dir=%s unit=%s" % (source, speed, requested_rpm, clamped_rpm, pwm, 'forward' if forward else 'reverse', self.section_name))

    def _get_calibrated_max_rpm(self):
        return max(self.calibration_map_points, key=lambda p: p['pwm'])['rpm'] if self.calibration_map_points else None

    def get_effective_max_rpm(self):
        return self._get_calibrated_max_rpm() or self.UNCALIBRATED_MAX_RPM

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

        rpm_buckets = {}
        for pwm_key in sorted(pwm_buckets.keys()):
            avg_rpm = sum(pwm_buckets[pwm_key]) / len(pwm_buckets[pwm_key])
            rpm_buckets.setdefault(round(avg_rpm, 1), []).append(pwm_key)

        normalized = [
            {'pwm': round(sum(rpm_buckets[rpm]) / len(rpm_buckets[rpm]), 6), 'rpm': rpm}
            for rpm in sorted(rpm_buckets)
        ]
        normalized.sort(key=lambda p: p['pwm'])

        if len(normalized) < self.CALIBRATION_MIN_POINTS:
            return None, 'insufficient_points'

        return normalized, None

    def set_calibration_map(self, payload, source='runtime'):
        if payload is None:
            self.calibration_map_points = []
            self.map_mode, self.map_fallback_reason = 'linear', 'map_missing'
            self._log_map_state('linear', 'map_missing')
            return False, 'map_missing'

        if not isinstance(payload, dict):
            return False, 'payload_type'

        points, reason = self._normalize_calibration_points(payload.get('points', []))
        if reason is not None:
            return False, reason

        self.calibration_map_points = points
        self.map_mode, self.map_fallback_reason = 'mapped', 'none'
        self._log_map_state(self.map_mode, '%s_loaded' % source)
        return True, None

    def get_calibration_map_payload(self): return None if not self.calibration_map_points else {'points': [dict(point) for point in self.calibration_map_points]}

    def rpm_to_pwm(self, rpm):
        if self.calibration_map_points and self.tachometer.has_tachometer():
            self.map_mode, self.map_fallback_reason = 'mapped', 'none'
            self._log_map_state('mapped', 'mapped')
            return self._rpm_to_pwm_mapped(rpm)
        reason = 'map_missing' if not self.calibration_map_points else 'tachometer_missing'
        self.map_mode = 'linear'
        self.map_fallback_reason = reason
        self._log_map_state('linear', reason)
        if rpm <= EPSILON:
            return 0.
        return self.pwm_min + (self.pwm_max - self.pwm_min) * min(rpm / self.get_effective_max_rpm(), 1.)

    def _rpm_to_pwm_mapped(self, rpm):
        points = self.calibration_map_points
        min_point = min(points, key=lambda p: p['rpm'])
        max_point = max(points, key=lambda p: p['rpm'])
        if rpm <= min_point['rpm']:
            return min_point['pwm']
        if rpm >= max_point['rpm']:
            return max_point['pwm']
        return min(points, key=lambda p: (abs(p['rpm'] - rpm), -p['pwm']))['pwm']

    def calibrate_pwm_rpm_map(self, point_count):
        if not self.tachometer.has_tachometer():
            raise MmuError("BLDC tachometer unavailable for calibration in [%s]" % self.section_name)

        step = 0. if point_count == 1 else (self.pwm_max - self.pwm_min) / float(point_count - 1)
        sweep_pwm = [self.pwm_max] if point_count == 1 else [self.pwm_min + (step * i) for i in range(point_count)]
        self.mmu.log_stepper("BLDC_CALIBRATE: initialize sweep points=%i sweep_pwm=%s unit=%s" % (point_count, repr(sweep_pwm), self.section_name))

        raw_points = []
        saved_enabled = self.tachometer.enabled
        saved_kick_start_time = self.kick_start_time
        self.tachometer.enabled = False
        self.kick_start_time = 0.
        self.mmu.log_stepper("BLDC_CALIBRATE: tach control suppressed unit=%s" % self.section_name)
        try:
            self.stop()
            self.mmu.movequeues_wait()
            self._safe_set_direction(True)

            for pwm in sweep_pwm:
                self._safe_set_pwm(pwm)
                self.mmu.movequeues_wait()
                self.reactor.pause(
                    self.reactor.monotonic()
                    + self.CALIBRATION_DEFAULT_SETTLE_S
                    + self.CALIBRATION_DEFAULT_SAMPLE_S
                )

                valid, reason = self.tachometer.is_tach_sample_valid()
                if not valid:
                    self.mmu.log_warning("Skipping BLDC calibration sample pwm=%.4f: %s (%s)" % (pwm, reason, self.section_name))
                    continue

                raw_points.append({'pwm': round(pwm, 6), 'rpm': round(self.tachometer.last_tach_rpm, 1)})
                self.mmu.log_stepper("BLDC_CALIBRATE: sample pwm=%.4f rpm=%.1f unit=%s" % (pwm, self.tachometer.last_tach_rpm, self.section_name))
        finally:
            self.tachometer.enabled = saved_enabled
            self.kick_start_time = saved_kick_start_time
            self.mmu.log_stepper("BLDC_CALIBRATE: tach control restored enabled=%s unit=%s" % (saved_enabled, self.section_name))
            self.stop()
            self.mmu.movequeues_wait()

        ok, reason = self.set_calibration_map({'points': raw_points}, source='calibration')
        if not ok:
            raise MmuError(
                "BLDC calibration failed for [%s]: %s (valid points=%d)"
                % (self.section_name, reason, len(raw_points))
            )
        return self.get_calibration_map_payload()

    def supports_gate(self, gate): return gate is not None and self.first_gate <= gate < self.first_gate + self.num_gates

    def _set_pwm_callback(self, print_time, value):
        log_time = print_time if print_time is not None else 0.
        if self.last_pwm <= EPSILON < value and self.kick_start_time > EPSILON:
            self.last_pwm = self.pwm_max
            self.mcu_pwm_pin.set_pwm(print_time, self.pwm_max)
            self.mmu.log_stepper(
                "BLDC_SET_PIN: kick req=%.4f applied=%.4f time=%.3f unit=%s"
                % (value, self.pwm_max, log_time, self.section_name)
            )
            return 'delay', self.kick_start_time
        if abs(value - self.last_pwm) < EPSILON:
            self.mmu.log_stepper("BLDC_SET_PIN: discard req=%.4f applied=%.4f unit=%s" % (value, self.last_pwm, self.section_name))
            return 'discard', 0.
        self.last_pwm = value
        self.mcu_pwm_pin.set_pwm(print_time, value)
        self.mmu.log_stepper(
            "BLDC_SET_PIN: pin=%s applied=%.4f time=%.3f unit=%s"
            % (str(getattr(self.mcu_pwm_pin, '_pin', 'unknown')), value, log_time, self.section_name)
        )
        return None

    def _set_dir_callback(self, print_time, value):
        log_time = print_time if print_time is not None else 0.
        ivalue = 1 if value else 0
        if self.last_dir == ivalue:
            return 'discard', 0.
        self.last_dir = ivalue
        self.mcu_dir_pin.set_digital(print_time, ivalue)
        self.mmu.log_stepper("BLDC_SET_PIN: pin=%s applied=%.4f time=%.3f unit=%s" % (str(getattr(self.mcu_dir_pin, '_pin', 'unknown')), ivalue, log_time, self.section_name))
        return None

    def _send_pin(self, mcu_pin, value, print_time):
        min_sched_time = mcu_pin.get_mcu().min_schedule_time()
        while True:
            if mcu_pin is self.mcu_pwm_pin:
                ret = self._set_pwm_callback(print_time, value)
            elif mcu_pin is self.mcu_dir_pin:
                ret = self._set_dir_callback(print_time, value)
            else:
                break
            if ret is None:
                break
            action, delay = ret
            if action == 'discard':
                break
            if action == 'delay':
                print_time += max(delay, min_sched_time)
                continue
            break

    def _floored_print_time(self, print_time=None):
        t = self._get_scheduled_print_time()
        return t if print_time is None or print_time < t else print_time

    def _safe_set_direction(self, forward, print_time=None):
        desired_dir = int(forward)
        if self.last_dir == desired_dir:
            return
        self._send_pin(self.mcu_dir_pin, desired_dir, self._floored_print_time(print_time))

    def _safe_set_pwm(self, value, print_time=None):
        if abs(value - self.last_pwm) < self.PWM_WRITE_MIN_DELTA:
            return
        if value > EPSILON:
            current_print_time = self.tachometer.get_current_print_time()
            if current_print_time - self._last_pwm_write_print_time < self.PWM_WRITE_MIN_INTERVAL_S - EPSILON:
                return
            self._last_pwm_write_print_time = current_print_time
        self._send_pin(self.mcu_pwm_pin, value, self._floored_print_time(print_time))

    def queue_trapzoid_move(self, move, axis_r, print_time, source):
        if move.accel_t <= EPSILON and move.cruise_t <= EPSILON and move.decel_t <= EPSILON:
            return
        direction = 1 if move.cruise_v * axis_r > 0. else -1
        accel_mm_s2 = move.accel * abs(axis_r)
        start_v = move.start_v * axis_r
        cruise_v = move.cruise_v * axis_r
        end_v = move.end_v * axis_r
        if move.accel_t > EPSILON:
            self.motion_queue.append((MotionTrapzoid(
                direction, start_v, cruise_v, accel_mm_s2,
                print_time=self._floored_print_time(print_time),
            ), source))
        if move.cruise_t > EPSILON:
            self.motion_queue.append((MotionDescriptor(
                cruise_v, True, self._floored_print_time(print_time + move.accel_t), direction,
            ), source))
        if move.decel_t > EPSILON:
            self.motion_queue.append((MotionTrapzoid(
                direction, cruise_v, end_v, accel_mm_s2,
                print_time=self._floored_print_time(print_time + move.accel_t + move.cruise_t),
            ), source))
        self.motion_state = self.MOTION_STATE_MOVING
        self._ensure_motion_timer()
        self.reactor.update_timer(self.motion_timer, self.reactor.NOW)

    def _get_scheduled_print_time(self):
        mcu = self.mcu_pwm_pin.get_mcu()
        return mcu.estimated_print_time(self.reactor.monotonic() + mcu.min_schedule_time() + 0.015)

    def _handle_connect(self):
        """Probe and resolve sync monitor on connect (extruder guaranteed registered)."""
        try:
            monitor = getattr(self.mmu, '_bldc_process_move_monitor', None)
            if monitor is None:
                monitor = ProcessMoveSyncMonitor(self.mmu)
                setattr(self.mmu, '_bldc_process_move_monitor', monitor)
            if not monitor.activate(self):
                raise self.config.error(
                    "sync_monitor failed to activate in [%s]" % (self.section_name)
                )
            monitor.deactivate(self)
            self.active_sync_monitor = monitor
            self.mmu.log_stepper("BLDC_SYNC: monitor probed and validated")
        except Exception as e:
            if 'error' in str(type(e).__name__).lower():
                raise
            raise self.config.error(
                "Failed to initialize sync monitor in [%s]: %s" % (self.section_name, str(e))
            )

    def _ensure_motion_timer(self):
        if self.motion_timer is None:
            self.motion_timer = self.reactor.register_timer(self._motion_timer_callback)

    def _dispatch_motion_descriptor(self, descriptor, source, print_time):
        speed_mm_s = descriptor.speed_mm_s
        self._log_sync_sample(source, 0., 0., speed_mm_s, speed_mm_s, 1., abs(speed_mm_s) > EPSILON)
        if abs(speed_mm_s) < EPSILON:
            if abs(self.last_pwm) > EPSILON:
                self.commanded_rpm = 0.
                self.commanded_source = source
                self.commanded_linear_speed = 0.
                self.last_effective_pwm = 0.
                self.tachometer.set_commanded(0., source)
                self._send_pin(self.mcu_pwm_pin, 0., self._floored_print_time(print_time))
            return True
        forward = self._map_forward_for_gate(speed_mm_s > 0., speed_mm_s)
        self._set_rpm(60. * abs(speed_mm_s) / self.rotation_distance, forward, source, abs(speed_mm_s), print_time)
        self.tachometer.enabled = descriptor.pid_enable
        self.mmu.log_stepper("BLDC_MOTION: source=%s time=%.3f unit=%s" % (source, print_time, self.section_name))
        return True

    def _get_motion_waketime(self, eventtime, current_print_time):
        if not self.motion_queue:
            return self.reactor.NEVER
        next_print_time = min(descriptor.print_time for descriptor, _ in self.motion_queue)
        delta_to_deadline = (next_print_time - self.motion_sample_time) - current_print_time
        if delta_to_deadline <= EPSILON:
            return eventtime + EPSILON
        return eventtime + min(self.motion_sample_time, delta_to_deadline)

    def _motion_timer_callback(self, eventtime):
        if self.motion_state != self.MOTION_STATE_MOVING:
            return self.reactor.NEVER

        current_print_time = self.mcu_pwm_pin.get_mcu().estimated_print_time(eventtime)

        self.motion_queue = [
            (descriptor, source) for descriptor, source in self.motion_queue
            if descriptor.print_time is not None and descriptor.print_time >= current_print_time - EPSILON
        ]

        if not self.motion_queue:
            if not self.sync_active:
                self.stop()
                return self.reactor.NEVER
            return self._get_motion_waketime(eventtime, current_print_time)

        descriptor, source = next(
            ((desc, src) for desc, src in self.motion_queue
             if desc.print_time <= current_print_time + self.motion_sample_time + EPSILON),
            (None, None))

        if descriptor is None or descriptor.print_time > current_print_time + EPSILON:
            return self._get_motion_waketime(eventtime, current_print_time)

        dispatch_print_time = max(descriptor.print_time, current_print_time)
        if descriptor.pwm is not None:
            t = self._floored_print_time(dispatch_print_time)
            dir_val = 1 if descriptor.direction > 0 else 0
            self._send_pin(self.mcu_dir_pin, dir_val, t)
            self._send_pin(self.mcu_pwm_pin, descriptor.pwm, t)
            self.mmu.log_stepper("BLDC_MOTION: pwm_direct pwm=%.4f dir=%d time=%.3f unit=%s" % (descriptor.pwm, dir_val, dispatch_print_time, self.section_name))
        else:
            self._dispatch_motion_descriptor(descriptor, source, dispatch_print_time)

        wake_delay = descriptor.advance(dispatch_print_time)
        if wake_delay is None:
            return self._get_motion_waketime(eventtime, current_print_time)
        if wake_delay >= INFINITY - EPSILON:
            self.motion_queue = []
            return self.reactor.NEVER
        return eventtime + wake_delay

    def _set_rpm(self, rpm, forward, source='move', linear_speed=0., print_time=None):
        requested_rpm = rpm
        rpm = max(0., min(rpm, self.get_effective_max_rpm()))
        t0 = self._floored_print_time(print_time)
        if rpm > EPSILON and self.commanded_rpm > EPSILON \
                and abs(rpm - self.commanded_rpm) > self.LARGE_SPEED_CHANGE_RPM:
            self.tachometer.reset_integral()
            self._last_pwm_write_print_time = 0.
        elif rpm > EPSILON >= self.commanded_rpm:
            self._last_pwm_write_print_time = 0.
        self.commanded_rpm = rpm
        self.commanded_source = source
        self.commanded_linear_speed = linear_speed
        self.tachometer.set_commanded(rpm, source)
        pwm = self.rpm_to_pwm(rpm)
        effective_pwm = self.tachometer.apply_control(pwm, source)
        self.last_effective_pwm = effective_pwm
        self._log_speed(source, linear_speed, requested_rpm, rpm, effective_pwm, forward)
        self._safe_set_direction(forward, t0)
        self._send_pin(self.mcu_pwm_pin, effective_pwm, t0)

    def set_speed(self, speed_mm_s, print_time=None, source='move'):
        self.mmu.log_stepper("BLDC_SET_SPEED: requested speed=%.3f mm/s source=%s unit=%s" % (speed_mm_s, source, self.section_name))
        if abs(speed_mm_s) < EPSILON:
            self.stop(print_time)
            return
        forward = self._map_forward_for_gate(speed_mm_s > 0., speed_mm_s)
        self._set_rpm(60. * abs(speed_mm_s) / self.rotation_distance, forward, source, abs(speed_mm_s), print_time)

    def _reset_motion(self, state, source):
        self.motion_state = state
        if self.motion_timer is not None:
            self.reactor.update_timer(self.motion_timer, self.reactor.NEVER)
        self.motion_queue = []
        self.commanded_rpm = 0.
        self.commanded_source = source
        self.commanded_linear_speed = 0.
        self.last_effective_pwm = 0.
        self._last_pwm_write_print_time = 0.
        self.tachometer.stop()
        self.tachometer.enabled = False

    def stop(self, print_time=None):
        self._reset_motion(self.MOTION_STATE_STOP, 'stop')
        if abs(self.last_pwm) > EPSILON:
            self._send_pin(self.mcu_pwm_pin, 0., self._floored_print_time(print_time))
        toolhead = self.printer.lookup_object('toolhead', None)
        if toolhead is not None and hasattr(toolhead, 'flush_step_generation'):
            toolhead.flush_step_generation()

    def brake_to_stop(self, print_time=None):
        applied_pwm = abs(self.last_pwm)
        if applied_pwm < self.BRAKE_MIN_ACTIVE_PWM or self.last_dir is None:
            self.stop(print_time)
            return

        brake_scale = min(1., max(0., applied_pwm / max(self.pwm_max, EPSILON)))
        brake_time = max(self.BRAKE_MIN_TIME_S, brake_scale * self.brake_max_time)
        reverse_dir = 0 if self.last_dir else 1

        self._reset_motion(self.MOTION_STATE_BRAKE, 'brake')
        t0 = self._floored_print_time(print_time)
        brake_descriptor = MotionPwmDirect(-1, self.brake_pwm, brake_time, t0)
        stop_descriptor = MotionStop(t0 + brake_time)
        self.motion_queue.append((brake_descriptor, 'brake'))
        self.motion_queue.append((stop_descriptor, 'brake'))
        self._send_pin(self.mcu_dir_pin, 0 if reverse_dir else 1, t0)
        self._send_pin(self.mcu_pwm_pin, self.brake_pwm, t0)
        self._ensure_motion_timer()
        self.reactor.update_timer(self.motion_timer, self.reactor.NOW)
        self.mmu.log_stepper(
            "BLDC_BRAKE: trigger_pwm=%.4f brake_pwm=%.4f brake_time=%.4f dir=%d unit=%s"
            % (applied_pwm, self.brake_pwm, brake_time, reverse_dir, self.section_name)
        )

    def start_move(self, dist, speed):
        speed = abs(speed)
        if dist == 0. or speed <= EPSILON:
            self.stop()
            return
        mapped_dist = self._map_distance_for_gate(dist)[0]
        direction = 1 if mapped_dist > 0. else -1
        target_speed = speed * direction
        start_time = self._floored_print_time()
        move_duration = abs(mapped_dist) / speed
        stop_time = self._floored_print_time(start_time + move_duration)

        # Queue fixed-duration move phases so standalone MMU_TEST_MOVE keeps BLDC active
        # for full move duration and stops deterministically at the end of the window.
        kick_time = min(self.kick_start_time, move_duration)
        if kick_time > EPSILON:
            self.motion_queue.append((
                MotionPwmDirect(direction, self.pwm_max, kick_time, start_time),
                'move',
            ))

        cruise_start_time = self._floored_print_time(start_time + kick_time)
        if move_duration - kick_time > EPSILON:
            self.motion_queue.append((
                MotionDescriptor(target_speed, True, cruise_start_time, direction),
                'move',
            ))

        self.motion_queue.append((MotionStop(stop_time), 'move'))
        self.motion_state = self.MOTION_STATE_MOVING
        self._ensure_motion_timer()
        self.reactor.update_timer(self.motion_timer, self.reactor.NOW)

    def set_rotation_distance(self, value):
        if value > 0.:
            self.rotation_distance = value

    def has_tachometer(self): return self.tachometer.has_tachometer()

    def get_rotation_distance(self): return self.rotation_distance

    def get_status(self, _eventtime):
        tach_status = self.tachometer.get_status()
        return {
            'active': abs(self.last_pwm) > EPSILON, 'pwm': self.last_pwm, 'dir': self.last_dir,
            'rotation_distance': self.rotation_distance,
            'tachometer_frequency': tach_status['tachometer_frequency'], 'tachometer_rpm': tach_status['tachometer_rpm'],
            'tachometer_fresh': tach_status['tachometer_fresh'], 'commanded_rpm': self.commanded_rpm,
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
        self.mmu.log_stepper("BLDC_SYNC: synced, starting BLDC and initializing sync state (unit=%s)" % self.section_name)
        self.sync_active = True
        self.last_sync_speed_log_eventtime = self.last_sync_sample_log_eventtime = None
        if self.active_sync_monitor is not None:
            self.active_sync_monitor.activate(self)

    def _handle_unsynced(self):
        self.mmu.log_stepper("BLDC_SYNC: unsynced, stopping BLDC and clearing sync state (unit=%s)" % self.section_name)
        self.sync_active = False
        self.last_sync_speed_log_eventtime = self.last_sync_sample_log_eventtime = None
        if self.active_sync_monitor is not None:
            self.active_sync_monitor.deactivate(self)
        if self.motion_state == self.MOTION_STATE_MOVING and self.motion_queue:
            self.mmu.log_stepper("BLDC_SYNC: draining queued motion before stop (unit=%s)" % self.section_name)
            return
        self.stop()

    def _handle_shutdown(self):
        if self.active_sync_monitor is not None:
            self.active_sync_monitor.deactivate(self)
        self.stop()

    def set_sync_enabled(self, enabled):
        (self._handle_synced if enabled else self._handle_unsynced)()
