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


class SyncMonitorBase:
    def activate(self, bldc):
        raise NotImplementedError()

    def deactivate(self, bldc):
        raise NotImplementedError()

    def update(self, eventtime):
        raise NotImplementedError()

    def is_active_source(self):
        raise NotImplementedError()


class ProcessMoveSyncMonitor(SyncMonitorBase):
    """Singleton monitor per MMU that owns process_move hook wrapping and move-derived sync."""

    PHASE_SAMPLE_TIME = 0.050

    def __init__(self, mmu):
        self.mmu = mmu
        self.hooked_extruder = None
        self.original_process_move = None
        self.hook_enabled = False
        self.active_bldc = None

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
            self.hook_enabled = True
            self.active_bldc = bldc
            return True

        original = process_move

        def wrapped_process_move(hooked_self, print_time, move, ea_index):
            original(print_time, move, ea_index)
            self._handle_process_move(print_time, move, ea_index)

        extruder._hh_bldc_original_process_move = original
        extruder._hh_bldc_process_move_owner = self
        extruder.process_move = MethodType(wrapped_process_move, extruder)
        self.hooked_extruder = extruder
        self.original_process_move = original
        self.hook_enabled = True
        self.active_bldc = bldc
        self.mmu.log_stepper("BLDC_PROCESS_MOVE: hook installed")
        return True

    def deactivate(self, bldc):
        if self.active_bldc is not bldc:
            return

        extruder = self.hooked_extruder
        if extruder is None:
            self.hook_enabled = False
            self.original_process_move = None
            self.active_bldc = None
            return

        if getattr(extruder, '_hh_bldc_process_move_owner', None) is self and self.original_process_move is not None:
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

    def update(self, eventtime):
        """No-op: process_move updates happen directly in the hook, no polling needed."""
        pass

    def is_active_source(self):
        return self.hook_enabled and self.active_bldc is not None

    def _phase_sample_times(self, phase_start_time, phase_end_time):
        """Yield sample times in (phase_start_time, phase_end_time], forcing phase_end_time."""
        if phase_end_time <= phase_start_time:
            return []

        sample_times = []
        sample_time = phase_start_time + self.PHASE_SAMPLE_TIME
        while sample_time < phase_end_time:
            sample_times.append(sample_time)
            sample_time += self.PHASE_SAMPLE_TIME
        sample_times.append(phase_end_time)
        return sample_times

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
        accel_end_time = print_time + move.accel_t
        decel_start_time = accel_end_time + move.cruise_t
        move_end_print_time = decel_start_time + move.decel_t
        bldc._log_process_move(print_time, start_speed, cruise_speed, end_speed, move)

        last_scheduled_speed = [None]

        def schedule_speed(speed, sample_print_time):
            if last_scheduled_speed[0] is not None and abs(speed - last_scheduled_speed[0]) < 1e-9:
                return
            bldc.set_speed(speed, print_time=sample_print_time, source='process_move_push')
            last_scheduled_speed[0] = speed

        # Phase 1 schedule at move start (non-zero only)
        first_speed = cruise_speed if abs(cruise_speed) != 0. else start_speed
        if abs(first_speed) != 0.:
            schedule_speed(first_speed, print_time)

        # Phase 2 accel window samples in (print_time, accel_end_time]
        if move.accel_t > 0.:
            for sample_print_time in self._phase_sample_times(print_time, accel_end_time):
                ratio = (sample_print_time - print_time) / move.accel_t
                sample_speed = start_speed + (cruise_speed - start_speed) * ratio
                schedule_speed(sample_speed, sample_print_time)

        # Phase 3 decel window samples in (decel_start_time, move_end_print_time]
        if move.decel_t > 0.:
            for sample_print_time in self._phase_sample_times(decel_start_time, move_end_print_time):
                ratio = (sample_print_time - decel_start_time) / move.decel_t
                sample_speed = cruise_speed + (end_speed - cruise_speed) * ratio
                schedule_speed(sample_speed, sample_print_time)

        # Schedule stop at move end
        bldc.stop(print_time=move_end_print_time)


class SampledSyncMonitor(SyncMonitorBase):
    """Singleton sampled-position fallback monitor per MMU.
    Owns its own cadence timer for push-driven speed updates."""

    SAMPLE_POLL_INTERVAL = 0.02  # Cadence for position sampling

    def __init__(self, mmu):
        self.mmu = mmu
        self.enabled = False
        self.timer = None
        self.last_pos = None
        self.last_time = None
        self.active_bldc = None

    def activate(self, bldc):
        self.active_bldc = bldc
        self.enabled = True
        self.last_pos = None
        self.last_time = None
        # Register cadence timer for push-driven updates
        if self.timer is None:
            self.timer = bldc.reactor.register_timer(self._timer_callback)
        bldc.reactor.update_timer(self.timer, bldc.reactor.NOW)
        return True

    def deactivate(self, bldc):
        if self.active_bldc is not bldc:
            return
        self.enabled = False
        self.active_bldc = None
        self.last_pos = None
        self.last_time = None
        # Cancel cadence timer
        if self.timer is not None:
            bldc.reactor.update_timer(self.timer, bldc.reactor.NEVER)

    def _timer_callback(self, eventtime):
        """Timer callback for sampling cadence."""
        if not self.is_active_source():
            return self.active_bldc.reactor.NEVER
        self.update(eventtime)
        return eventtime + self.SAMPLE_POLL_INTERVAL

    def update(self, eventtime):
        if not self.is_active_source():
            return

        bldc = self.active_bldc
        pos, sample_time, speed_scale, source = self._get_sample(eventtime)
        if self.last_time is None:
            self.last_pos = pos
            self.last_time = sample_time
            return

        dt = sample_time - self.last_time
        de = pos - self.last_pos
        self.last_pos = pos
        self.last_time = sample_time

        if dt <= 0.:
            return

        speed = de / dt
        selected_speed = speed * speed_scale
        moving = abs(selected_speed) != 0.

        bldc._log_sync_sample(source, dt, de, speed, selected_speed, speed_scale, moving)
        if not moving:
            bldc.stop()
            return

        bldc.set_speed(selected_speed, print_time=sample_time, source='sync')

    def is_active_source(self):
        return self.enabled and self.active_bldc is not None

    def _get_sample(self, eventtime):
        try:
            mcu = self.mmu.printer.lookup_object('mcu')
            print_time = mcu.estimated_print_time(eventtime)
            extruder = self.mmu.toolhead.get_extruder()
            if extruder is not None:
                return extruder.find_past_position(print_time), print_time, 1., 'past_position'
        except Exception:
            pass

        extruder_stepper = getattr(self.mmu, 'mmu_extruder_stepper', None)
        if extruder_stepper is not None:
            try:
                mcu_pos = extruder_stepper.stepper.get_mcu_position()
                step_dist = extruder_stepper.stepper.get_step_dist()
                pos = extruder_stepper.stepper.mcu_to_commanded_position(mcu_pos)
                return pos, eventtime, step_dist * self.active_bldc.sync_speed_factor, 'mcu_position'
            except Exception:
                pass

        try:
            return self.mmu.toolhead.get_position()[3], eventtime, 1., 'toolhead_position'
        except Exception:
            return 0., eventtime, 1., 'fallback_zero'


class MmuGearBldc:
    SYNC_SPEED_LOG_INTERVAL = 0.2
    MONITOR_PROCESS_MOVE = 'process_move'
    MONITOR_SAMPLED = 'sampled'
    SYNC_MONITOR_OPTIONS = [MONITOR_PROCESS_MOVE, MONITOR_SAMPLED]

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

        self.last_pwm = 0.
        self.last_dir = None
        self.section_name = config.get_name()

        self.dir_change_deadtime = config.getfloat('dir_change_deadtime', 0.02, minval=0.)
        self.sync_speed_factor = config.getfloat('sync_speed_factor', 0.694, above=0.)
        self._sync_monitor_kind = config.getchoice('sync_monitor', {o: o for o in self.SYNC_MONITOR_OPTIONS}, self.MONITOR_PROCESS_MOVE)

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
        if self.last_sync_sample_log_eventtime is not None:
            if eventtime - self.last_sync_sample_log_eventtime < self.SYNC_SPEED_LOG_INTERVAL:
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
            if abs(value - self.last_pwm) < 0.0001:
                self.mmu.log_stepper("BLDC_SET_PIN: discard.")
                return 'discard', 0.
            self.last_pwm = value
            self.mcu_pwm_pin.set_pwm(print_time, value)
            self.mmu.log_stepper("BLDC_SET_PIN: pin=%s value=%.4f time=%.3f" % (mcu_pin._pin, value, print_time or 0.))
            return None

        if mcu_pin is self.mcu_dir_pin:
            ivalue = 1 if value else 0
            if self.last_dir == ivalue:
                return 'discard', 0.
            self.last_dir = ivalue
            self.mcu_dir_pin.set_digital(print_time, ivalue)
            return None

        return 'discard', 0.

    def _send_pin(self, mcu_pin, value, print_time=None):
        self.gcrqs[mcu_pin.get_mcu()].send_async_request((mcu_pin, value), print_time)

    def _safe_set_direction(self, forward, print_time=None):
        desired_dir = 1 if forward else 0
        if self.last_dir is None or self.last_dir != desired_dir:
            self._send_pin(self.mcu_dir_pin, desired_dir, print_time)
            return

    def _get_print_time(self, mcu):
        systime = self.reactor.monotonic()
        return mcu.estimated_print_time(systime + mcu.min_schedule_time())

    def get_current_extruder(self):
        try:
            return self.mmu.toolhead.get_extruder()
        except Exception:
            return self.printer.lookup_object(getattr(self.mmu, 'extruder_name', 'extruder'), None)

    def _get_shared_monitor(self, kind):
        """Get or create per-MMU per-kind sync monitor singleton cached on mmu object."""
        if kind == self.MONITOR_PROCESS_MOVE:
            attr = '_bldc_process_move_monitor'
            monitor_class = ProcessMoveSyncMonitor
        elif kind == self.MONITOR_SAMPLED:
            attr = '_bldc_sampled_monitor'
            monitor_class = SampledSyncMonitor
        else:
            raise ValueError("Unknown sync monitor kind: %s" % kind)

        monitor = getattr(self.mmu, attr, None)
        if monitor is None:
            monitor = monitor_class(self.mmu)
            setattr(self.mmu, attr, monitor)
        return monitor

    def _handle_connect(self):
        """Probe and resolve sync monitor on connect (extruder guaranteed registered)."""
        try:
            monitor = self._get_shared_monitor(self._sync_monitor_kind)
            # Probe: activate and immediately deactivate to validate compatibility
            probe_ok = monitor.activate(self)
            monitor.deactivate(self)

            if not probe_ok:
                raise self.config.error(
                    "'sync_monitor=%s' failed to activate in [%s]" % (self._sync_monitor_kind, self.section_name)
                )
            self.active_sync_monitor = monitor
            self.mmu.log_stepper("BLDC_SYNC: monitor '%s' probed and validated" % self._sync_monitor_kind)
        except Exception as e:
            if 'error' in str(type(e).__name__).lower():
                raise
            raise self.config.error(
                "Failed to initialize sync monitor '%s' in [%s]: %s" % (self._sync_monitor_kind, self.section_name, str(e))
            )

    def _activate_sync_monitor(self):
        """Activate the probe-validated monitor for real sync motion."""
        if self.active_sync_monitor is not None:
            self.active_sync_monitor.activate(self)

    def _deactivate_sync_monitor(self):
        if self.active_sync_monitor is None:
            return
        self.active_sync_monitor.deactivate(self)

    def _rpm_to_pwm(self, rpm):
        if rpm <= 0.:
            return 0.
        ratio = min(rpm / self.max_rpm, 1.)
        return self.pwm_min + (self.pwm_max - self.pwm_min) * ratio

    def _log_speed(self, source, linear_speed, requested_rpm, clamped_rpm, pwm, forward):
        eventtime = self.reactor.monotonic()
        if source == 'sync':
            if self.last_sync_speed_log_eventtime is not None:
                if eventtime - self.last_sync_speed_log_eventtime < self.SYNC_SPEED_LOG_INTERVAL:
                    return
            self.last_sync_speed_log_eventtime = eventtime
        direction = 'forward' if forward else 'reverse'
        self.mmu.log_stepper(
            "BLDC_SPEED: source=%s linear_speed=%.3f requested_rpm=%.1f clamped_rpm=%.1f pwm=%.4f dir=%s unit=%s"
            % (source, linear_speed, requested_rpm, clamped_rpm, pwm, direction, self.section_name)
        )

    def _set_rpm(self, rpm, forward, source='move', linear_speed=0., print_time=None):
        requested_rpm = rpm
        rpm = max(0., min(rpm, self.max_rpm))
        pwm = self._rpm_to_pwm(rpm)
        self._log_speed(source, linear_speed, requested_rpm, rpm, pwm, forward)
        t0 = print_time if print_time is not None else self._get_print_time(self.mcu_pwm_pin.get_mcu())
        self._safe_set_direction(forward, t0)
        if self.last_dir != (1 if forward else 0):
            t0 += self.dir_change_deadtime
        self._send_pin(self.mcu_pwm_pin, pwm, t0)

    def set_speed(self, speed_mm_s, print_time=None, source='move'):
        if speed_mm_s == 0.:
            self.stop(print_time=print_time)
            return
        forward = self._map_forward_for_gate(speed_mm_s > 0., speed_mm_s)
        rpm = 60. * abs(speed_mm_s) / self.rotation_distance
        self._set_rpm(rpm, forward, source=source, linear_speed=abs(speed_mm_s), print_time=print_time)

    def stop(self, print_time=None):
        if self.last_pwm > 0.:
            self._send_pin(self.mcu_pwm_pin, 0., print_time)

    def start_move(self, dist, speed):
        if dist == 0.:
            self.stop()
            return
        speed = abs(speed)
        if speed <= 0.:
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
            'active': self.last_pwm > 0.,
            'pwm': self.last_pwm,
            'dir': self.last_dir,
            'rotation_distance': self.rotation_distance,
            'gate_start': self.first_gate,
            'gate_end': self.first_gate + self.num_gates - 1,
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