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
import traceback
from .mmu_shared import MmuError
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

        bldc.queue_trapzoid_move(move, axis_r, print_time, 'process_move_push')

class BldcTachometer:
    """Own tachometer sampling and PID PWM correction for BLDC."""

    TACH_VALID_MAX_AGE = 0.5
    CONTROL_DEADBAND_RPM = 50.
    CONTROL_MIN_RPM = 150.
    CONTROL_DT_MAX_SAMPLE_FACTOR = 2.

    PID_PARAM_BASE = 255.

    def __init__(self, bldc, config):
        self.mmu = bldc.mmu
        self.section_name = bldc.section_name
        self.reactor = bldc.reactor
        self.mcu = bldc.mcu
        self.mcu_pwm_pin = bldc.mcu_pwm_pin
        self.bldc = bldc

        self.pwm_min = config.getfloat('pwm_min', 0.85, minval=0., maxval=1.)
        self.pwm_max = config.getfloat('pwm_max', 1.0, minval=0., maxval=1.)
        self.tachometer_ppr = config.getint('tachometer_ppr', 9, minval=1)
        self.tachometer_sample_time = config.getfloat('tachometer_sample_time', 0.1, above=0.)
        self.tachometer_stale_time = config.getfloat('tachometer_stale_time', self.tachometer_sample_time * 3., above=0.)
        self.control_kp = config.getfloat('tachometer_control_kp', 12.0, minval=0.) / self.PID_PARAM_BASE
        self.control_ki = config.getfloat('tachometer_control_ki', 200.0, minval=0.) / self.PID_PARAM_BASE
        self.control_max_delta_pwm = config.getfloat('tachometer_control_max_delta_pwm', 0.20, minval=0., maxval=1.)

        tachometer_pin = config.get('tachometer_pin', None)
        self.tachometer = None
        if tachometer_pin is not None:
            self.tachometer = pulse_counter.MCU_counter(
                bldc.printer, tachometer_pin, self.tachometer_sample_time,
                config.getfloat('tachometer_poll_interval', 0.001, above=0.0009)
            )

        self.commanded_rpm = 0.
        self.commanded_source = None
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

    def apply_control(self, pwm, source):
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
            return
        delta_time = count_time - self._tach_last_count_time
        frequency = (count - self._tach_last_count) / delta_time if delta_time > 0. else 0.
        control_dt = delta_time if delta_time > 0. else (max(0., time - self.last_tach_eventtime) if self.last_tach_eventtime is not None else 0.)
        if control_dt <= 0.:
            control_dt = self.tachometer_sample_time
        control_dt = min(control_dt, self.tachometer_sample_time * self.CONTROL_DT_MAX_SAMPLE_FACTOR)
        self._tach_last_count = count
        self._tach_last_count_time = count_time
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
        else:
            if abs(error_rpm) <= self.CONTROL_DEADBAND_RPM:
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
    SCHEDULE_LEAD_S = 0.25

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
        self.last_dir = None
        self.section_name = config.get_name()
        self.commanded_rpm = self.commanded_linear_speed = 0.
        self.commanded_source = None
        self.calibration_map_points = []
        self.map_mode = 'linear'
        self.map_fallback_reason = 'map_missing'

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

        self.mcu = self.mcu_pwm_pin.get_mcu()

        if self.mcu is not self.mcu_dir_pin.get_mcu():
            raise config.error(
                "'pwm_pin' and 'dir_pin' in [%s] must be on the same mcu"
                % config.get_name()
            )

        self.min_schedule_time = self.mcu.min_schedule_time() + self.SCHEDULE_LEAD_S
        self.motion_sample_time = self.mcu.min_schedule_time()
        self._gcrqs_native_mode = False
        self.gcrqs = self._create_gcrqs(config)

        self.tachometer = BldcTachometer(self, config)

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
            return False, 'map_missing'

        if not isinstance(payload, dict):
            return False, 'payload_type'

        points, reason = self._normalize_calibration_points(payload.get('points', []))
        if reason is not None:
            return False, reason

        self.calibration_map_points = points
        self.map_mode, self.map_fallback_reason = 'mapped', 'none'
        return True, None

    def get_calibration_map_payload(self):
        return None if not self.calibration_map_points else {'points': [dict(point) for point in self.calibration_map_points]}

    def rpm_to_pwm(self, rpm):
        if self.calibration_map_points and self.tachometer.has_tachometer():
            self.map_mode, self.map_fallback_reason = 'mapped', 'none'
            return self._rpm_to_pwm_mapped(rpm)
        reason = 'map_missing' if not self.calibration_map_points else 'tachometer_missing'
        self.map_mode = 'linear'
        self.map_fallback_reason = reason
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

        raw_points = []
        saved_enabled = self.tachometer.enabled
        saved_kick_start_time = self.kick_start_time
        self.tachometer.enabled = False
        self.kick_start_time = 0.
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
        finally:
            self.tachometer.enabled = saved_enabled
            self.kick_start_time = saved_kick_start_time
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

    def _create_gcrqs(self, config):
        # Prefer native output_pin queue, then reuse espooler fallback queue for Kalico compatibility.
        try:
            from .. import output_pin
            if hasattr(output_pin, 'GCodeRequestQueue'):
                self._gcrqs_native_mode = True
                return output_pin.GCodeRequestQueue(config, self.mcu, self._queue_set_pin)
        except Exception:
            pass

        from ..mmu_espooler import GCodeRequestQueue as FallbackGCodeRequestQueue
        self._gcrqs_native_mode = False
        return FallbackGCodeRequestQueue(config, self.mcu, self._queue_set_pin)

    def _queue_set_pin(self, print_time, action):
        def _pin_name(mcu_pin):
            return {
                self.mcu_pwm_pin: 'pwm',
                self.mcu_dir_pin: 'dir',
            }.get(mcu_pin, 'unknown')

        def _queue_discard():
            return 'discard', 0.

        def _queue_delay(wait_s):
            wait_s = max(0., wait_s)
            self.mmu.log_stepper(
                "BLDC_SET_PIN: delay pin=%s value=%.4f wait=%.6f print_time=%.6f"
                % (_pin_name(mcu_pin), value, wait_s, print_time)
            )
            if self._gcrqs_native_mode:
                return 'reschedule', print_time + wait_s
            return 'delay', wait_s

        mcu_pin, value = action
        if mcu_pin is self.mcu_pwm_pin:
            if abs(value - self.last_pwm) < self.PWM_WRITE_MIN_DELTA:
                self.mmu.log_stepper(
                    "BLDC_SET_PIN: discard pin=pwm value=%.4f print_time=%.6f reason=small_delta"
                    % (value, print_time)
                )
                return _queue_discard()
            if value > EPSILON:
                elapsed = print_time - self._last_pwm_write_print_time
                if elapsed < self.PWM_WRITE_MIN_INTERVAL_S - EPSILON:
                    return _queue_delay(self.PWM_WRITE_MIN_INTERVAL_S - elapsed)
            msg = self._set_pwm_callback(print_time, value)
            if msg == 'discard':
                self.mmu.log_stepper(
                    "BLDC_SET_PIN: discard pin=pwm value=%.4f print_time=%.6f reason=duplicate"
                    % (value, print_time)
                )
                return _queue_discard()
            if value > EPSILON:
                self._last_pwm_write_print_time = print_time
        elif mcu_pin is self.mcu_dir_pin:
            msg = self._set_dir_callback(print_time, value)
            if msg == 'discard':
                self.mmu.log_stepper(
                    "BLDC_SET_PIN: discard pin=dir value=%.4f print_time=%.6f reason=duplicate"
                    % (value, print_time)
                )
                return _queue_discard()
        else:
            self.mmu.log_stepper(
                "BLDC_SET_PIN: discard pin=unknown value=%.4f print_time=%.6f reason=bad_pin"
                % (value, print_time)
            )
            return _queue_discard()

        return None

    def _set_pwm_callback(self, print_time, value):
        if abs(value - self.last_pwm) < EPSILON:
            return 'discard'
        self.mcu_pwm_pin.set_pwm(print_time, value)
        self.last_pwm = value
        self.mmu.log_stepper("BLDC_SET_PIN: pwm value=%.4f print_time=%.6f" % (value, print_time))
        return ''

    def _set_dir_callback(self, print_time, value):
        ivalue = 1 if value else 0
        if self.last_dir == ivalue:
            return 'discard'
        self.mcu_dir_pin.set_digital(print_time, ivalue)
        self.last_dir = ivalue
        self.mmu.log_stepper("BLDC_SET_PIN: dir value=%d print_time=%.6f" % (ivalue, print_time))
        return ''

    def _send_pin(self, mcu_pin, value, print_time):
        t = self._get_scheduled_print_time() + self.min_schedule_time
        floored_print_time = t if print_time is None or print_time < t else print_time

        if mcu_pin is self.mcu_pwm_pin or mcu_pin is self.mcu_dir_pin:
            self.gcrqs.send_async_request((mcu_pin, value), floored_print_time)

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
        if not self._motion_timer_running and self.motion_state == self.MOTION_STATE_MOVING:
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

    def _motion_timer_callback(self, eventtime):
        if self.motion_state != self.MOTION_STATE_MOVING:
            self._motion_timer_running = False
            return self.reactor.NEVER

        # Base dispatch on the live clock, not the reactor-provided eventtime, so writes stay ahead
        # of the real MCU clock even after the gear+extruder quiesce stalls the reactor.
        current_print_time = self._get_scheduled_print_time() + self.min_schedule_time

        # Prune descriptors whose full time window has elapsed
        kept_motion_queue = []
        for d, src in self.motion_queue:
            expire_time = d.print_time + d.duration
            if expire_time >= current_print_time - EPSILON:
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
                self._queue_pwm_if_changed(0., None)
                self._reset_motion_command('sync_queue_drained')
            return self.reactor.NEVER

        # Filter to the next-tick window (early send: ensure all candidates within MCU schedule constraint)
        candidates = [
            (d, src) for d, src in self.motion_queue
            if d.print_time <= current_print_time
        ]

        if not candidates:
            return eventtime + self.motion_sample_time

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
                if self._queue_pwm_if_changed(0., dispatch_time):
                    self._log_descriptor(desc, dispatch_time)
            elif desc.pwm is not None:
                dir_val = 1 if desc.direction > 0 else 0
                self._send_pin(self.mcu_dir_pin, dir_val, dispatch_time)
                if self._queue_pwm_if_changed(desc.pwm, dispatch_time):
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
                    effective_pwm = self.tachometer.apply_control(pwm, src)
                    self._send_pin(self.mcu_dir_pin, int(forward), dispatch_time)
                    if self._queue_pwm_if_changed(effective_pwm, dispatch_time):
                        self._log_descriptor(desc, dispatch_time)

        # Check if motion should stop (MotionStop was in batch and is last)
        if batch and isinstance(batch[-1][0], MotionStop):
            return self.reactor.NEVER

        return eventtime + self.motion_sample_time

    def _reset_motion_command(self, source):
        self.commanded_rpm = 0.
        self.commanded_source = source
        self.commanded_linear_speed = 0.
        self.last_effective_pwm = 0.
        self._last_pwm_write_print_time = 0.
        self.tachometer.stop()
        self.tachometer.enabled = False

    def _reset_motion(self, state, source):
        self.motion_state = state
        self._stop_motion_timer()
        self.motion_queue = []
        self._reset_motion_command(source)

    def stop(self, print_time=None):
        self._reset_motion(self.MOTION_STATE_STOP, 'stop')
        if abs(self.last_pwm) > EPSILON:
            self._send_pin(self.mcu_pwm_pin, 0., print_time)
        toolhead = self.printer.lookup_object('toolhead', None)
        if toolhead is not None and hasattr(toolhead, 'flush_step_generation'):
            toolhead.flush_step_generation()

    def brake_to_stop(self):
        print_time = self._get_scheduled_print_time()
        applied_pwm = abs(self.last_pwm)
        if applied_pwm < self.BRAKE_MIN_ACTIVE_PWM or self.last_dir is None:
            self.stop(print_time)
            return

        brake_scale = min(1., max(0., applied_pwm / max(self.pwm_max, EPSILON)))
        brake_time = max(self.BRAKE_MIN_TIME_S, brake_scale * self.brake_max_time)
        reverse_dir = 0 if self.last_dir else 1

        self._reset_motion(self.MOTION_STATE_BRAKE, 'brake')
        brake_descriptor = MotionPwmDirect(print_time, self.brake_pwm, brake_time, -1)
        stop_descriptor = MotionStop(print_time + brake_time)
        self.motion_queue.append((brake_descriptor, 'brake'))
        self.motion_queue.append((stop_descriptor, 'brake'))
        self._send_pin(self.mcu_dir_pin, 0 if reverse_dir else 1, print_time)
        self._send_pin(self.mcu_pwm_pin, self.brake_pwm, print_time)

    def start_move(self, dist, speed):
        requested_speed = speed
        speed = abs(speed)
        if (dist is not None and dist == 0.) or speed <= EPSILON:
            self.stop()
            return
        # A new deliberate move supersedes any prior standalone move. Drop leftover descriptors
        # (notably a not-yet-dispatched MotionStop from the previous move whose print_time is now
        # in the past) so they cannot preempt this move on the next motion-timer tick.
        self.motion_queue = []
        start_time = self._get_scheduled_print_time() + self.min_schedule_time

        if dist is None:
            # Open-ended/event-driven mode. Direction is encoded in speed sign.
            requested_forward = requested_speed >= 0.
            direction = 1 if self._map_forward_for_gate(requested_forward, requested_speed) else -1
            target_speed = speed * direction
            kick_time = self.kick_start_time
            if kick_time > EPSILON:
                self.motion_queue.append((
                    MotionPwmDirect(start_time, self.pwm_max, kick_time, direction),
                    'move',
                ))
            cruise_start_time = start_time + kick_time
            self.motion_queue.append((
                MotionDescriptor(cruise_start_time, target_speed, True, direction, duration=INFINITY),
                'move',
            ))
        else:
            mapped_dist = self._map_distance_for_gate(dist)[0]
            direction = 1 if mapped_dist > 0. else -1
            target_speed = speed * direction
            move_duration = abs(mapped_dist) / speed

            # Queue fixed-duration move phases so standalone MMU_TEST_MOVE keeps BLDC active
            # for full move duration and stops deterministically at the end of the window.
            kick_time = min(self.kick_start_time, move_duration)
            if kick_time > EPSILON:
                self.motion_queue.append((
                    MotionPwmDirect(start_time, self.pwm_max, kick_time, direction),
                    'move',
                ))

            cruise_start_time = start_time + kick_time
            cruise_duration = move_duration - kick_time
            if cruise_duration > EPSILON:
                self.motion_queue.append((
                    MotionDescriptor(cruise_start_time, target_speed, True, direction, duration=cruise_duration),
                    'move',
                ))
            self.motion_queue.append((MotionStop(start_time + move_duration), 'move'))
        self.motion_state = self.MOTION_STATE_MOVING
        self._start_motion_timer()

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
