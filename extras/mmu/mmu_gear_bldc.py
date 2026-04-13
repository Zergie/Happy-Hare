# Happy Hare MMU Software
#
# BLDC gear controller for MMU gear motion replacement.
#
# Copyright (C) 2022-2026  moggieuk#6538 (discord)
#                          moggieuk@hotmail.com
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#

from .. import output_pin
from .. import mmu_espooler


class MmuGearBldc:
    SYNC_POLL_INTERVAL = 0.02
    SYNC_SPEED_LOG_INTERVAL = 0.2

    def __init__(self, config, mmu, first_gate=0, num_gates=1):
        self.config = config
        self.mmu = mmu
        self.first_gate = first_gate
        self.num_gates = num_gates
        self.printer = config.get_printer()
        self.reactor = self.printer.get_reactor()

        self.gcrqs = {}
        self.sync_timer = None
        self.sync_active = False
        self.last_sync_eventtime = None
        self.last_sync_extruder_pos = None
        self.manual_hold_active = False
        self.manual_hold_until = None
        self.last_sync_speed_log_eventtime = None
        self.last_sync_sample_log_eventtime = None
        self.sync_monitor = SyncMonitor(self)

        self.last_pwm = 0.
        self.last_dir = None
        self.section_name = config.get_name()

        self.dir_change_deadtime = config.getfloat('dir_change_deadtime', 0.02, minval=0.)
        self.sync_min_speed = config.getfloat('sync_min_speed', 0.05, minval=0.)
        self.sync_speed_factor = config.getfloat('sync_speed_factor', 0.694, above=0.)

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
        effective_dist, map_value, gate = self._map_distance_for_gate(requested_dist, gate)
        return effective_dist >= 0.

    def _log_sync_sample(self, eventtime, source, dt, de, raw_speed, selected_speed, speed_scale, moving):
        if self.last_sync_sample_log_eventtime is not None:
            if eventtime - self.last_sync_sample_log_eventtime < self.SYNC_SPEED_LOG_INTERVAL:
                return
        self.last_sync_sample_log_eventtime = eventtime
        self.mmu.log_stepper(
            "BLDC_SYNC_SAMPLE: source=%s dt=%.4f de=%.4f raw_speed=%.3f selected_speed=%.3f scale=%.3f moving=%d unit=%s"
            % (source, dt, de, raw_speed, selected_speed, speed_scale, 1 if moving else 0, self.section_name)
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
            self.mmu.log_stepper("BLDC_SEND_PIN: pin=%s value=%.4f time=%.3f called=%s" % (self.mcu_dir_pin._pin, desired_dir, print_time or 0., "safe_set_direction"))
            self._send_pin(self.mcu_dir_pin, desired_dir, print_time)
            return

    def _get_print_time(self, mcu):
        systime = self.reactor.monotonic()
        return mcu.estimated_print_time(systime + mcu.min_schedule_time())

    def _rpm_to_pwm(self, rpm):
        if rpm <= 0.:
            return 0.
        ratio = min(rpm / self.max_rpm, 1.)
        return self.pwm_min + (self.pwm_max - self.pwm_min) * ratio

    def _log_speed(self, source, linear_speed, requested_rpm, clamped_rpm, pwm, forward, eventtime=None):
        if source == 'sync' and eventtime is not None:
            if self.last_sync_speed_log_eventtime is not None:
                if eventtime - self.last_sync_speed_log_eventtime < self.SYNC_SPEED_LOG_INTERVAL:
                    return
            self.last_sync_speed_log_eventtime = eventtime
        direction = 'forward' if forward else 'reverse'
        self.mmu.log_stepper(
            "BLDC_SPEED: source=%s linear_speed=%.3f requested_rpm=%.1f clamped_rpm=%.1f pwm=%.4f dir=%s unit=%s"
            % (source, linear_speed, requested_rpm, clamped_rpm, pwm, direction, self.section_name)
        )

    def _set_target(self, rpm, forward, source='move', linear_speed=0., eventtime=None):
        requested_rpm = rpm
        rpm = max(0., min(rpm, self.max_rpm))
        pwm = self._rpm_to_pwm(rpm)
        self._log_speed(source, linear_speed, requested_rpm, rpm, pwm, forward, eventtime)
        t0 = self._get_print_time(self.mcu_pwm_pin.get_mcu())
        self._safe_set_direction(forward, t0)
        if self.last_dir != (1 if forward else 0):
            t0 += self.dir_change_deadtime
        self.mmu.log_stepper("BLDC_SEND_PIN: pin=%s value=%.4f time=%.3f called=%s" % (self.mcu_pwm_pin._pin, pwm, t0 or 0., "set_target"))
        self._send_pin(self.mcu_pwm_pin, pwm, t0)

    def stop(self):
        if self.last_pwm > 0.:
            self.mmu.log_stepper("BLDC_SEND_PIN: pin=%s value=%.4f time=%.3f called=%s" % (self.mcu_pwm_pin._pin, 0, 0., "stop"))
            self._send_pin(self.mcu_pwm_pin, 0.)

    def start_move(self, dist, speed):
        if dist == 0.:
            self.stop()
            return
        speed = abs(speed)
        if speed <= 0.:
            self.stop()
            return
        effective_dist, map_value, gate = self._map_distance_for_gate(dist)
        forward = effective_dist > 0.
        rpm = 60. * speed / self.rotation_distance
        self._set_target(rpm, forward, source='move', linear_speed=speed)

    def start_move_hold(self, dist, speed, hold_seconds=None):
        self.start_move(dist, speed)
        self.manual_hold_active = True
        if hold_seconds is None:
            self.manual_hold_until = None
        else:
            hold_seconds = max(0., hold_seconds)
            self.manual_hold_until = self.reactor.monotonic() + hold_seconds

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
        self.mmu.log_stepper("BLDC_SYNC: synced, starting BLDC and initializing sync state")
        self.sync_active = True
        self.last_sync_eventtime = None
        self.last_sync_extruder_pos = None
        self.last_sync_speed_log_eventtime = None
        self.last_sync_sample_log_eventtime = None
        self.sync_monitor.watch(True)
        if self.sync_timer is None:
            self.sync_timer = self.reactor.register_timer(self._sync_update)
        self.reactor.update_timer(self.sync_timer, self.reactor.NOW)

    def _handle_unsynced(self):
        self.mmu.log_stepper("BLDC_SYNC: unsynced, stopping BLDC and clearing sync state")
        self.sync_active = False
        self.last_sync_eventtime = None
        self.last_sync_extruder_pos = None
        self.last_sync_speed_log_eventtime = None
        self.last_sync_sample_log_eventtime = None
        self.manual_hold_active = False
        self.manual_hold_until = None
        self.sync_monitor.watch(False)
        self.stop()
        if self.sync_timer is not None:
            self.reactor.update_timer(self.sync_timer, self.reactor.NEVER)

    def _handle_shutdown(self):
        self.stop()

    def set_sync_enabled(self, enabled):
        if enabled:
            self._handle_synced()
        else:
            self._handle_unsynced()

    def _sync_update(self, eventtime):
        if not self.sync_active:
            return self.reactor.NEVER

        if self.manual_hold_active:
            if self.manual_hold_until is None or eventtime < self.manual_hold_until:
                return eventtime + self.SYNC_POLL_INTERVAL
            self.manual_hold_active = False
            self.manual_hold_until = None

        sample = self.sync_monitor.update(eventtime)
        if sample is None:
            return eventtime + self.SYNC_POLL_INTERVAL

        dt, de, speed_scale, source = sample

        if dt <= 0.:
            return eventtime + self.SYNC_POLL_INTERVAL

        speed = de / dt
        selected_speed = speed * speed_scale
        moving = abs(selected_speed) > self.sync_min_speed

        if not moving:
            self._log_sync_sample(eventtime, source, dt, de, speed, selected_speed, speed_scale, False)
            self.stop()
            return eventtime + self.SYNC_POLL_INTERVAL

        self._log_sync_sample(eventtime, source, dt, de, speed, selected_speed, speed_scale, True)

        rpm = 60. * abs(selected_speed) / self.rotation_distance
        self._set_target(
            rpm,
            self._map_forward_for_gate(selected_speed >= 0., selected_speed),
            source='sync',
            linear_speed=abs(selected_speed),
            eventtime=eventtime,
        )

        return eventtime + self.SYNC_POLL_INTERVAL


class SyncMonitor:
    def __init__(self, bldc):
        self.bldc = bldc
        self.enabled = False
        self.last_pos = None
        self.last_time = None

    def watch(self, enabled):
        self.enabled = enabled
        self.last_pos = None
        self.last_time = None

    def _get_sample(self, eventtime):
        try:
            mcu = self.bldc.printer.lookup_object('mcu')
            print_time = mcu.estimated_print_time(eventtime)
            extruder = self.bldc.mmu.toolhead.get_extruder()
            if extruder is not None:
                return extruder.find_past_position(print_time), print_time, 1., 'past_position'
        except Exception:
            pass

        extruder_stepper = getattr(self.bldc.mmu, 'mmu_extruder_stepper', None)
        if extruder_stepper is not None:
            try:
                mcu_pos = extruder_stepper.stepper.get_mcu_position()
                step_dist = extruder_stepper.stepper.get_step_dist()
                pos = extruder_stepper.stepper.mcu_to_commanded_position(mcu_pos)
                return pos, eventtime, step_dist * self.bldc.sync_speed_factor, 'mcu_position'
            except Exception:
                pass

        try:
            return self.bldc.mmu.toolhead.get_position()[3], eventtime, 1., 'toolhead_position'
        except Exception:
            return 0., eventtime, 1., 'fallback_zero'

    def update(self, eventtime):
        pos, sample_time, speed_scale, source = self._get_sample(eventtime)
        if self.last_time is None:
            self.last_pos = pos
            self.last_time = sample_time
            return None

        dt = sample_time - self.last_time
        de = pos - self.last_pos
        self.last_pos = pos
        self.last_time = sample_time
        return dt, de, speed_scale, source