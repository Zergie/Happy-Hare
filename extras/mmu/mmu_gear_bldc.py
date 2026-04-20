# Happy Hare MMU Software
#
# BLDC gear controller for MMU gear motion replacement.
#
# Copyright (C) 2022-2026  moggieuk#6538 (discord)
#                          moggieuk@hotmail.com
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#

import inspect
from types import MethodType

from .. import output_pin
from .. import mmu_espooler
from .. import pulse_counter


class SyncMonitorBase:
    def activate(self, bldc):
        raise NotImplementedError()

    def deactivate(self, bldc):
        raise NotImplementedError()

class ProcessMoveSyncMonitor(SyncMonitorBase):
    """Singleton monitor per MMU that owns process_move hook wrapping and move-derived sync."""

    PHASE_SAMPLE_TIME = 0.100
    EMIT_LEAD_TIME = 0.030

    def __init__(self, mmu):
        self.mmu = mmu
        self.hooked_extruder = None
        self.original_process_move = None
        self.hook_enabled = False
        self.active_bldc = None
        self.current_move = None
        self.move_active = False
        self.last_emitted_sample_time = None
        self.move_generation = 0
        self.timer = None

    def _attach_bldc(self, bldc):
        self.hook_enabled = True
        self.active_bldc = bldc
        if self.timer is None:
            self.timer = bldc.reactor.register_timer(self._timer_callback)
        bldc.reactor.update_timer(self.timer, bldc.reactor.NOW)

    def activate(self, bldc):
        if self.hook_enabled and self.active_bldc is bldc:
            return True

        extruder = bldc.get_current_extruder()
        if extruder is None:
            self.mmu.log_stepper("BLDC_PROCESS_MOVE: no extruder")
            return False

        process_move = getattr(extruder, 'process_move', None)
        if process_move is None:
            self.mmu.log_stepper("BLDC_PROCESS_MOVE: process_move missing")
            return False

        params = list(inspect.signature(process_move).parameters)
        if params != ['print_time', 'move', 'ea_index']:
            self.mmu.log_stepper("BLDC_PROCESS_MOVE: bad signature=%s" % (','.join(params),))
            return False

        owner = getattr(extruder, '_hh_bldc_process_move_owner', None)
        if owner is not None and owner is not self:
            self.mmu.log_stepper("BLDC_PROCESS_MOVE: foreign hook present")
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

        if self.timer is not None:
            bldc.reactor.update_timer(self.timer, bldc.reactor.NEVER)

        self._reset_move_state()

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

        self.hooked_extruder = None
        self.original_process_move = None
        self.hook_enabled = False
        self.active_bldc = None

    def _reset_move_state(self):
        self.current_move = None
        self.move_active = False
        self.last_emitted_sample_time = None

    def _compute_speed_at_time(self, sample_print_time, move):
        move_start_time = move['move_start_time']
        move_end_time = move['move_end_time']
        if sample_print_time < move_start_time - 1e-9 or sample_print_time > move_end_time + 1e-9:
            return None, 'out_of_window'

        start_v = move['start_v']
        cruise_v = move['cruise_v']
        end_v = move['end_v']
        accel_t = move['accel_t']
        cruise_t = move['cruise_t']
        decel_t = move['decel_t']
        accel_end_time = move['accel_end_time']
        decel_start_time = move['decel_start_time']

        if sample_print_time <= move_start_time + 1e-9:
            return start_v, 'start'
        if sample_print_time >= move_end_time - 1e-9:
            return end_v, 'end'

        if accel_t > 0. and sample_print_time < accel_end_time:
            ratio = (sample_print_time - move_start_time) / accel_t
            ratio = min(1., max(0., ratio))
            return start_v + (cruise_v - start_v) * ratio, 'accel'

        if cruise_t > 0. and sample_print_time < decel_start_time:
            return cruise_v, 'cruise'

        if decel_t > 0.:
            ratio = (sample_print_time - decel_start_time) / decel_t
            ratio = min(1., max(0., ratio))
            return cruise_v + (end_v - cruise_v) * ratio, 'decel'

        return end_v, 'fallback'

    def _timer_callback(self, eventtime):
        reactor = self.mmu.printer.get_reactor()
        bldc = self.active_bldc
        if not self.hook_enabled or bldc is None or not bldc.sync_active:
            return reactor.NEVER
        if not self.move_active or self.current_move is None:
            return reactor.NEVER

        move = self.current_move
        if move.get('generation') != self.move_generation:
            return reactor.NEVER

        emit_print_time = bldc.mcu_pwm_pin.get_mcu().estimated_print_time(eventtime) + self.EMIT_LEAD_TIME
        move_start_time = move['move_start_time']
        move_end_time = move['move_end_time']

        if emit_print_time > move_end_time + 1e-9:
            if self.last_emitted_sample_time is None or self.last_emitted_sample_time < move_end_time - 1e-9:
                terminal_speed, _ = self._compute_speed_at_time(move_end_time, move)
                if terminal_speed is not None:
                    self.mmu.log_stepper(
                        "BLDC_PROCESS_MOVE: late terminal sample emit=%.3f end=%.3f unit=%s"
                        % (emit_print_time, move_end_time, bldc.section_name)
                    )
                    bldc.set_speed(terminal_speed, move_end_time, source='process_move_push')
                    self.last_emitted_sample_time = move_end_time
            self.mmu.log_stepper(
                "BLDC_PROCESS_MOVE: window expired before next sample emit=%.3f end=%.3f unit=%s"
                % (emit_print_time, move_end_time, bldc.section_name)
            )
            self._reset_move_state()
            return reactor.NEVER

        if emit_print_time < move_start_time - 1e-9:
            return bldc._estimate_systime_from_print_time(move_start_time - self.EMIT_LEAD_TIME)

        if self.last_emitted_sample_time is None:
            next_sample_time = move_start_time
        else:
            next_sample_time = self.last_emitted_sample_time + self.PHASE_SAMPLE_TIME
            if next_sample_time > emit_print_time + 1e-9:
                return bldc._estimate_systime_from_print_time(next_sample_time - self.EMIT_LEAD_TIME)

        latest_index = int(max(0., (emit_print_time - move_start_time)) / self.PHASE_SAMPLE_TIME)
        latest_sample_time = move_start_time + latest_index * self.PHASE_SAMPLE_TIME
        sample_print_time = max(next_sample_time, latest_sample_time)
        if sample_print_time > move_end_time + 1e-9:
            self._reset_move_state()
            return reactor.NEVER

        dropped_count = int(max(0., (sample_print_time - next_sample_time)) / self.PHASE_SAMPLE_TIME)
        if dropped_count > 0:
            self.mmu.log_stepper(
                "BLDC_PROCESS_MOVE: dropped stale samples=%d sample=%.3f emit=%.3f unit=%s"
                % (dropped_count, sample_print_time, emit_print_time, bldc.section_name)
            )

        sample_speed, phase = self._compute_speed_at_time(sample_print_time, move)
        if sample_speed is None:
            self._reset_move_state()
            return reactor.NEVER

        bldc.set_speed(sample_speed, sample_print_time, source='process_move_push')
        self.last_emitted_sample_time = sample_print_time

        if sample_print_time >= move_end_time - 1e-9:
            self.mmu.log_stepper(
                "BLDC_PROCESS_MOVE: move window complete sample=%.3f unit=%s"
                % (sample_print_time, bldc.section_name)
            )
            self._reset_move_state()
            return reactor.NEVER

        next_print_time = min(move_end_time, sample_print_time + self.PHASE_SAMPLE_TIME)
        if phase == 'start' and next_print_time <= move_start_time + 1e-9:
            next_print_time = min(move_end_time, move_start_time + self.PHASE_SAMPLE_TIME)
        return bldc._estimate_systime_from_print_time(next_print_time - self.EMIT_LEAD_TIME)

    def _handle_process_move(self, print_time, move, ea_index):
        bldc = self.active_bldc
        if bldc is None or not bldc.sync_active:
            return
        if ea_index < 3 or ea_index >= len(move.axes_r):
            return

        axis_r = move.axes_r[ea_index]
        if axis_r == 0.:
            return

        start_speed = move.start_v * axis_r
        cruise_speed = move.cruise_v * axis_r
        end_speed = move.end_v * axis_r
        move_duration = move.accel_t + move.cruise_t + move.decel_t
        if move_duration <= 0.:
            bldc.set_speed(end_speed, print_time, source='process_move_push')
            return

        accel_end_time = print_time + move.accel_t
        decel_start_time = accel_end_time + move.cruise_t
        move_end_print_time = decel_start_time + move.decel_t
        bldc._log_process_move(print_time, start_speed, cruise_speed, end_speed, move)
        previous_move = self.current_move
        if previous_move is not None and self.move_active:
            prev_end = previous_move['move_end_time']
            if print_time < prev_end - 1e-9:
                self.mmu.log_stepper(
                    "BLDC_PROCESS_MOVE: overlap replace old_start=%.3f old_end=%.3f new_start=%.3f unit=%s"
                    % (previous_move['move_start_time'], prev_end, print_time, bldc.section_name)
                )

        self.move_generation += 1
        self.current_move = {
            'generation': self.move_generation,
            'move_start_time': print_time,
            'start_v': start_speed,
            'cruise_v': cruise_speed,
            'end_v': end_speed,
            'accel_t': move.accel_t,
            'cruise_t': move.cruise_t,
            'decel_t': move.decel_t,
            'accel_end_time': accel_end_time,
            'decel_start_time': decel_start_time,
            'move_end_time': move_end_print_time,
        }
        self.move_active = True
        self.last_emitted_sample_time = None

        self.mmu.log_stepper(
            "BLDC_PROCESS_MOVE: captured window start=%.3f end=%.3f generation=%d unit=%s"
            % (print_time, move_end_print_time, self.move_generation, bldc.section_name)
        )
        if self.timer is not None:
            bldc.reactor.update_timer(self.timer, bldc.reactor.NOW)


class MmuGearBldc:
    SYNC_SPEED_LOG_INTERVAL = 0.2
    TACH_LOG_INTERVAL = 0.5
    CONTROL_LOG_INTERVAL = 0.2
    ZERO_EPSILON = 1e-6
    CONTROL_DEADBAND_RPM = 100.
    CONTROL_MIN_RPM = 150.

    def __init__(self, config, mmu, first_gate=0, num_gates=1):
        self.config = config
        self.mmu = mmu
        self.first_gate = first_gate
        self.num_gates = num_gates
        self.printer = config.get_printer()
        self.reactor = self.printer.get_reactor()

        self.gcrqs = {}
        self.sync_active = False
        self.last_sync_speed_log_eventtime = None
        self.last_sync_sample_log_eventtime = None
        self.last_tach_log_eventtime = None
        self.last_control_log_eventtime = None
        self.last_control_log_reason = None

        self.last_req_pwm = 0.
        self.last_pwm = 0.
        self.last_effective_pwm = 0.
        self.last_dir = None
        self.section_name = config.get_name()
        self.commanded_rpm = 0.
        self.commanded_source = None
        self.commanded_linear_speed = 0.
        self.last_tach_freq = 0.
        self.last_tach_rpm = 0.
        self.last_tach_eventtime = None

        self.kick_start_time = config.getfloat('kick_start_time', 0.1, minval=0.)
        self.sync_speed_factor = config.getfloat('sync_speed_factor', 0.694, above=0.)
        self.tachometer_pin = config.get('tachometer_pin', None)
        self.tachometer_ppr = config.getint('tachometer_ppr', 9, minval=1)
        self.tachometer_sample_time = config.getfloat('tachometer_sample_time', 0.1, above=0.)
        self.tachometer_poll_interval = config.getfloat('tachometer_poll_interval', 0.00025, above=0.)

        self.tachometer_stale_time = config.getfloat('tachometer_stale_time', self.tachometer_sample_time * 3., above=0.)
        self.tachometer = None
        self.control_enabled = config.getboolean('tachometer_control_enabled', False)
        self.control_kp = config.getfloat('tachometer_control_kp', 1.0, minval=0.)
        self.control_max_delta_pwm = config.getfloat('tachometer_control_max_delta_pwm', 0.05, minval=0., maxval=1.)
        self.control_correction_pwm = 0.
        self.control_enable_after_print_time = 0.
        self.control_reason = 'disabled'
        self.last_tach_error_rpm = 0.

        self.pwm_min = config.getfloat('pwm_min', 0.85, minval=0., maxval=1.)
        self.pwm_max = config.getfloat('pwm_max', 1.0, minval=0., maxval=1.)
        if self.pwm_min > self.pwm_max:
            raise config.error("'pwm_min' cannot be greater than 'pwm_max' in [%s]" % config.get_name())

        self.max_rpm = config.getfloat('max_rpm', 6000., above=0.)
        self.rotation_distance = config.getfloat('rotation_distance', 1.0, above=0.)
        self.direction_map = self._load_direction_map(config)

        self.hardware_pwm = config.getboolean('hardware_pwm', False)
        self.cycle_time = config.getfloat('cycle_time', 0.00005, above=0.)

        ppins = self.printer.lookup_object('pins')

        self.mcu_dir_pin = ppins.setup_pin('digital_out', config.get('dir_pin'))
        self.mcu_dir_pin.setup_max_duration(0.)
        self.mcu_dir_pin.setup_start_value(0., 0.)

        self.mcu_pwm_pin = ppins.setup_pin('pwm', config.get('pwm_pin'))
        self.mcu_pwm_pin.setup_max_duration(0.)
        self.mcu_pwm_pin.setup_cycle_time(self.cycle_time, self.hardware_pwm)
        self.mcu_pwm_pin.setup_start_value(0., 0.)

        if self.tachometer_pin is not None:
            self.tachometer = pulse_counter.MCU_counter(
                self.printer, self.tachometer_pin, self.tachometer_sample_time, self.tachometer_poll_interval
            )
            self.tachometer.setup_callback(self._handle_tachometer)
            self._tach_last_count = None
            self._tach_last_count_time = None

        self._kick_uses_repeat = hasattr(output_pin, 'GCodeRequestQueue')
        self._setup_queue(self.mcu_dir_pin)
        self._setup_queue(self.mcu_pwm_pin)

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

    def _get_selected_gate(self):
        return getattr(self.mmu, 'gate_selected', None)

    def _get_map_value_for_gate(self, gate):
        if gate is None:
            return 0
        if not self.supports_gate(gate):
            return 0
        local_gate = gate - self.first_gate
        if local_gate < 0 or local_gate >= len(self.direction_map):
            return 0
        return self.direction_map[local_gate]

    def _map_distance_for_gate(self, requested_dist, gate=None):
        gate = self._get_selected_gate() if gate is None else gate
        map_value = self._get_map_value_for_gate(gate)
        effective_dist = -requested_dist if map_value else requested_dist
        return effective_dist, map_value, gate

    def _map_forward_for_gate(self, requested_forward, requested_speed):
        gate = self._get_selected_gate()
        requested_dist = abs(requested_speed) if requested_forward else -abs(requested_speed)
        effective_dist, _, gate = self._map_distance_for_gate(requested_dist, gate)
        return effective_dist >= 0.

    def _log_sync_sample(self, source, dt, de, raw_speed, selected_speed, speed_scale, moving):
        eventtime = self.reactor.monotonic()
        if self.last_sync_sample_log_eventtime is not None \
                and eventtime - self.last_sync_sample_log_eventtime < self.SYNC_SPEED_LOG_INTERVAL:
            return
        self.last_sync_sample_log_eventtime = eventtime
        self.mmu.log_stepper(
            "BLDC_SYNC_SAMPLE: source=%s dt=%.4f de=%.4f raw_speed=%.3f selected_speed=%.3f scale=%.3f moving=%d unit=%s"
            % (source, dt, de, raw_speed, selected_speed, speed_scale, 1 if moving else 0, self.section_name)
        )

    def _log_process_move(self, print_time, start_speed, cruise_speed, end_speed, move):
        self.mmu.log_stepper(
            "BLDC_PROCESS_MOVE: print_time=%.3f start=%.3f cruise=%.3f end=%.3f accel_t=%.4f cruise_t=%.4f decel_t=%.4f unit=%s"
            % (print_time, start_speed, cruise_speed, end_speed, move.accel_t, move.cruise_t, move.decel_t, self.section_name)
        )

    def _set_control_state(self, reason, correction_pwm=0., error_rpm=None):
        self.control_reason = reason
        self.control_correction_pwm = correction_pwm
        if error_rpm is not None:
            self.last_tach_error_rpm = error_rpm
        self._log_control(reason, self.last_req_pwm, self.last_pwm)

    def _is_control_eligible(self):
        if not self.control_enabled:
            return False, 'disabled'
        if self.tachometer is None:
            return False, 'no_tachometer'
        if not self.sync_active:
            return False, 'sync_inactive'
        if self.commanded_source == 'stop' or self.commanded_rpm <= self.ZERO_EPSILON:
            return False, 'stopped'
        if self._get_current_print_time() < self.control_enable_after_print_time:
            return False, 'startup_inhibit'
        if self.commanded_rpm < self.CONTROL_MIN_RPM:
            return False, 'below_min_rpm'
        if not self._has_fresh_tachometer():
            return False, 'tach_stale'
        return True, 'active'

    def _log_control(self, source, base_pwm, applied_pwm):
        eventtime = self.reactor.monotonic()
        reason_changed = self.control_reason != self.last_control_log_reason
        time_elapsed = self.last_control_log_eventtime is None \
            or eventtime - self.last_control_log_eventtime >= self.CONTROL_LOG_INTERVAL
        if not reason_changed and not time_elapsed:
            return
        self.last_control_log_eventtime = eventtime
        self.last_control_log_reason = self.control_reason
        self.mmu.log_stepper(
            "BLDC_CONTROL: source=%s reason=%s error_rpm=%.1f base_pwm=%.4f correction_pwm=%.4f applied_pwm=%.4f unit=%s"
            % (source, self.control_reason, self.last_tach_error_rpm, base_pwm, self.control_correction_pwm, applied_pwm, self.section_name)
        )

    def _apply_control(self, pwm, source):
        if pwm <= self.ZERO_EPSILON:
            self._set_control_state('zero_pwm', 0., 0.)
            return 0.
        eligible, reason = self._is_control_eligible()
        if not eligible:
            self._set_control_state(reason, 0.)
            self._log_control(source, pwm, pwm)
            return pwm
        applied_pwm = min(self.pwm_max, max(self.pwm_min, pwm + self.control_correction_pwm))
        self._set_control_state('active', self.control_correction_pwm, self.last_tach_error_rpm)
        self._log_control(source, pwm, applied_pwm)
        return applied_pwm

    def _handle_tachometer(self, time, count, count_time):
        if self._tach_last_count is None:
            self._tach_last_count = count
            self._tach_last_count_time = count_time
            return
        delta_count = count - self._tach_last_count
        delta_time = count_time - self._tach_last_count_time
        frequency = delta_count / delta_time if delta_time > 0. else 0.
        self._tach_last_count = count
        self._tach_last_count_time = count_time
        self.last_tach_eventtime = time
        self.last_tach_freq = frequency
        self.last_tach_rpm = frequency * 30. / self.tachometer_ppr
        eligible, reason = self._is_control_eligible()
        if not eligible:
            self._set_control_state(reason, 0., self.commanded_rpm - self.last_tach_rpm)
        else:
            error_rpm = self.commanded_rpm - self.last_tach_rpm
            if abs(error_rpm) <= self.CONTROL_DEADBAND_RPM:
                self._set_control_state('deadband', 0., error_rpm)
            else:
                pwm_span = self.pwm_max - self.pwm_min
                correction_pwm = self.control_kp * (error_rpm / self.max_rpm) * pwm_span
                correction_pwm = max(-self.control_max_delta_pwm, min(self.control_max_delta_pwm, correction_pwm))
                self._set_control_state('active', correction_pwm, error_rpm)
        self._log_tachometer()

    def _has_fresh_tachometer(self):
        if self.tachometer is None or self.last_tach_eventtime is None:
            return False
        current_print_time = self.mcu_pwm_pin.get_mcu().estimated_print_time(self.reactor.monotonic())
        return current_print_time - self.last_tach_eventtime <= self.tachometer_stale_time

    def _log_tachometer(self):
        if self.tachometer is None:
            return
        eventtime = self.reactor.monotonic()
        if self.last_tach_log_eventtime is not None \
                and eventtime - self.last_tach_log_eventtime < self.TACH_LOG_INTERVAL:
            return
        self.last_tach_log_eventtime = eventtime
        if self.commanded_rpm < 100 and self.last_tach_rpm < 100:
            return
        self.mmu.log_stepper(
            "BLDC_TACH: freq=%.3f rpm=%.1f unit=%s"
            % (self.last_tach_freq, self.last_tach_rpm, self.section_name)
        )

    def supports_gate(self, gate):
        return gate is not None and self.first_gate <= gate < self.first_gate + self.num_gates

    def _setup_queue(self, mcu_pin):
        mcu = mcu_pin.get_mcu()
        if mcu in self.gcrqs:
            return
        if hasattr(output_pin, 'GCodeRequestQueue'):
            self.gcrqs[mcu] = output_pin.GCodeRequestQueue(self.config, mcu, self._set_pin)
        else:
            self.gcrqs[mcu] = mmu_espooler.GCodeRequestQueue(self.config, mcu, self._set_pin)

    def _set_pin(self, print_time, req):
        mcu_pin, value = req
        if mcu_pin is self.mcu_pwm_pin:
            self.last_req_pwm = value
            if value > self.ZERO_EPSILON and self.kick_start_time > self.ZERO_EPSILON and self.last_pwm <= self.ZERO_EPSILON:
                self.last_pwm = self.pwm_max
                self.mcu_pwm_pin.set_pwm(print_time, self.pwm_max)
                self.mmu.log_stepper(
                    "BLDC_SET_PIN: kick req=%.4f applied=%.4f time=%.3f unit=%s"
                    % (value, self.pwm_max, print_time or 0., self.section_name)
                )
                if self._kick_uses_repeat:
                    return 'repeat', print_time + self.kick_start_time
                return 'delay', self.kick_start_time
            if abs(value - self.last_pwm) < self.ZERO_EPSILON:
                self.mmu.log_stepper(
                    "BLDC_SET_PIN: discard req=%.4f applied=%.4f unit=%s"
                    % (value, self.last_pwm, self.section_name)
                )
                return 'discard', 0.
            self.last_pwm = value
            self.mcu_pwm_pin.set_pwm(print_time, value)
            self.mmu.log_stepper(
                "BLDC_SET_PIN: pin=%s applied=%.4f time=%.3f unit=%s"
                % (mcu_pin._pin, value, print_time or 0., self.section_name)
            )
            return None

        if mcu_pin is self.mcu_dir_pin:
            ivalue = 1 if value else 0
            if self.last_dir == ivalue:
                return 'discard', 0.
            self.last_dir = ivalue
            self.mcu_dir_pin.set_digital(print_time, ivalue)
            self.mmu.log_stepper(
                "BLDC_SET_PIN: pin=%s applied=%.4f time=%.3f unit=%s"
                % (mcu_pin._pin, ivalue, print_time or 0., self.section_name)
            )
            return None

        return 'discard', 0.

    def _send_pin(self, mcu_pin, value, print_time=None):
        self.gcrqs[mcu_pin.get_mcu()].send_async_request((mcu_pin, value), print_time)

    def _safe_set_direction(self, forward, print_time=None):
        desired_dir = int(forward)
        if self.last_dir != desired_dir:
            self._send_pin(self.mcu_dir_pin, desired_dir, print_time)

    def _get_print_time(self, mcu):
        return mcu.estimated_print_time(self.reactor.monotonic() + mcu.min_schedule_time())

    def _get_current_print_time(self):
        return self.mcu_pwm_pin.get_mcu().estimated_print_time(self.reactor.monotonic())

    def _estimate_systime_from_print_time(self, print_time):
        mcu = self.mcu_pwm_pin.get_mcu()
        clocksync = getattr(mcu, '_clocksync', None)
        if clocksync is None or not hasattr(clocksync, 'estimate_clock_systime'):
            return self.reactor.NOW
        reqclock = mcu.print_time_to_clock(max(0., print_time))
        return max(self.reactor.NOW, clocksync.estimate_clock_systime(reqclock))

    def get_current_extruder(self):
        try:
            return self.mmu.toolhead.get_extruder()
        except Exception:
            return self.printer.lookup_object(getattr(self.mmu, 'extruder_name', 'extruder'), None)

    def _get_shared_monitor(self):
        monitor = getattr(self.mmu, '_bldc_process_move_monitor', None)
        if monitor is None:
            monitor = ProcessMoveSyncMonitor(self.mmu)
            setattr(self.mmu, '_bldc_process_move_monitor', monitor)
        return monitor

    def _handle_connect(self):
        """Probe and resolve sync monitor on connect (extruder guaranteed registered)."""
        try:
            monitor = self._get_shared_monitor()
            # Probe: activate and immediately deactivate to validate compatibility
            probe_ok = monitor.activate(self)
            monitor.deactivate(self)

            if not probe_ok:
                raise self.config.error(
                    "sync_monitor failed to activate in [%s]" % (self.section_name)
                )
            self.active_sync_monitor = monitor
            self.mmu.log_stepper("BLDC_SYNC: monitor probed and validated")
        except Exception as e:
            if 'error' in str(type(e).__name__).lower():
                raise
            raise self.config.error(
                "Failed to initialize sync monitor in [%s]: %s" % (self.section_name, str(e))
            )

    def _activate_sync_monitor(self):
        """Activate the probe-validated monitor for real sync motion."""
        if self.active_sync_monitor is not None:
            self.active_sync_monitor.activate(self)

    def _deactivate_sync_monitor(self):
        if self.active_sync_monitor is None:
            return
        self.active_sync_monitor.deactivate(self)

    def _log_speed(self, source, speed, requested_rpm, clamped_rpm, pwm, forward):
        eventtime = self.reactor.monotonic()
        if source == 'sync':
            if self.last_sync_speed_log_eventtime is not None:
                if eventtime - self.last_sync_speed_log_eventtime < self.SYNC_SPEED_LOG_INTERVAL:
                    return
            self.last_sync_speed_log_eventtime = eventtime
        direction = 'forward' if forward else 'reverse'
        self.mmu.log_stepper(
            "BLDC_SPEED: source=%s speed=%.3f requested_rpm=%.1f clamped_rpm=%.1f pwm=%.4f dir=%s unit=%s"
            % (source, speed, requested_rpm, clamped_rpm, pwm, direction, self.section_name)
        )

    def _rpm_to_pwm(self, rpm):
        if rpm <= self.ZERO_EPSILON:
            return 0.
        ratio = min(rpm / self.max_rpm, 1.)
        return self.pwm_min + (self.pwm_max - self.pwm_min) * ratio

    def _set_rpm(self, rpm, forward, source='move', linear_speed=0., print_time=None):
        requested_rpm = rpm
        rpm = max(0., min(rpm, self.max_rpm))
        t0 = print_time if print_time is not None else self._get_print_time(self.mcu_pwm_pin.get_mcu())
        if self.commanded_rpm <= self.ZERO_EPSILON and rpm > self.ZERO_EPSILON:
            self.control_enable_after_print_time = t0 + self.kick_start_time + self.tachometer_sample_time
        self.commanded_rpm = rpm
        self.commanded_source = source
        self.commanded_linear_speed = linear_speed
        pwm = self._rpm_to_pwm(rpm)
        effective_pwm = self._apply_control(pwm, source)
        self.last_effective_pwm = effective_pwm
        self._log_speed(source, linear_speed, requested_rpm, rpm, effective_pwm, forward)
        self._safe_set_direction(forward, t0)
        self._send_pin(self.mcu_pwm_pin, effective_pwm, t0)

    def set_speed(self, speed_mm_s, print_time=None, source='move'):
        self.mmu.log_stepper(
            "BLDC_SET_SPEED: requested speed=%.3f mm/s source=%s unit=%s"
            % (speed_mm_s, source, self.section_name)
        )
        if abs(speed_mm_s) < self.ZERO_EPSILON:
            self.stop(print_time=print_time)
            return
        forward = self._map_forward_for_gate(speed_mm_s > 0., speed_mm_s)
        rpm = 60. * abs(speed_mm_s) / self.rotation_distance
        self._set_rpm(rpm, forward, source=source, linear_speed=abs(speed_mm_s), print_time=print_time)

    def stop(self, print_time=None):
        self.commanded_rpm = 0.
        self.commanded_source = 'stop'
        self.commanded_linear_speed = 0.
        self.control_enable_after_print_time = 0.
        self.last_effective_pwm = 0.
        self._set_control_state('stopped', 0., 0.)
        if abs(self.last_pwm) > self.ZERO_EPSILON:
            self._send_pin(self.mcu_pwm_pin, 0., print_time)

    def start_move(self, dist, speed):
        if dist == 0.:
            self.stop()
            return
        speed = abs(speed)
        if speed <= self.ZERO_EPSILON:
            self.stop()
            return
        effective_dist, _, _ = self._map_distance_for_gate(dist)
        signed_speed = speed if effective_dist > 0. else -speed
        self.set_speed(signed_speed, source='move')

    def set_rotation_distance(self, value):
        if value > 0.:
            self.rotation_distance = value

    def get_rotation_distance(self):
        return self.rotation_distance

    def get_status(self, _eventtime):
        return {
            'active': abs(self.last_pwm) > self.ZERO_EPSILON,
            'pwm': self.last_pwm,
            'dir': self.last_dir,
            'rotation_distance': self.rotation_distance,
            'gate_start': self.first_gate,
            'gate_end': self.first_gate + self.num_gates - 1,
            'tachometer_enabled': self.tachometer is not None,
            'tachometer_ppr': self.tachometer_ppr,
            'tachometer_frequency': self.last_tach_freq,
            'tachometer_rpm': self.last_tach_rpm,
            'tachometer_fresh': self._has_fresh_tachometer(),
            'commanded_rpm': self.commanded_rpm,
            'tachometer_error_rpm': self.commanded_rpm - self.last_tach_rpm,
            'control_enabled': self.control_enabled,
            'control_reason': self.control_reason,
            'control_correction_pwm': self.control_correction_pwm,
            'effective_pwm': self.last_effective_pwm,
        }

    def _handle_synced(self):
        self.mmu.log_stepper("BLDC_SYNC: synced, starting BLDC and initializing sync state (unit=%s)" % self.section_name)
        self.sync_active = True
        self.last_sync_speed_log_eventtime = None
        self.last_sync_sample_log_eventtime = None
        self._activate_sync_monitor()

    def _handle_unsynced(self):
        self.mmu.log_stepper("BLDC_SYNC: unsynced, stopping BLDC and clearing sync state (unit=%s)" % self.section_name)
        self.sync_active = False
        self.last_sync_speed_log_eventtime = None
        self.last_sync_sample_log_eventtime = None
        self._deactivate_sync_monitor()
        self.stop()

    def _handle_shutdown(self):
        self._deactivate_sync_monitor()
        self.stop()

    def set_sync_enabled(self, enabled):
        if enabled:
            self._handle_synced()
        else:
            self._handle_unsynced()