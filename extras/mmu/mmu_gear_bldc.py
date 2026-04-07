# Happy Hare MMU Software
#
# BLDC gear controller for MMU gear motion replacement.
#
# Copyright (C) 2022-2026  moggieuk#6538 (discord)
#                          moggieuk@hotmail.com
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#

from .. import output_pin, pulse_counter
from ..mmu_espooler import GCodeRequestQueue as FallbackGCodeRequestQueue


class MmuGearBldc:
    SYNC_POLL_INTERVAL = 0.02

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

        self.last_pwm = 0.
        self.last_dir = None
        self.pending_dir = None

        self.dir_change_deadtime = config.getfloat('dir_change_deadtime', 0.02, minval=0.)
        self.sync_min_speed = config.getfloat('sync_min_speed', 0.05, minval=0.)
        self.sync_confidence_timeout = config.getfloat('sync_confidence_timeout', 0.5, above=0.)

        self.pwm_min = config.getfloat('pwm_min', 0.85, minval=0., maxval=1.)
        self.pwm_max = config.getfloat('pwm_max', 1.0, minval=0., maxval=1.)
        if self.pwm_min > self.pwm_max:
            raise config.error("'pwm_min' cannot be greater than 'pwm_max' in [%s]" % config.get_name())

        self.max_rpm = config.getfloat('max_rpm', 6000., above=0.)
        self.rotation_distance = config.getfloat('rotation_distance', 1.0, above=0.)

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

        self.tachometer = None
        self.tach_ppr = config.getint('tach_ppr', 9, minval=1)
        tach_pin = config.get('tach_pin', None)
        if tach_pin is not None:
            poll_time = config.getfloat('tach_poll_interval', 0.0015, above=0.)
            sample_time = config.getfloat('tach_sample_time', 1., above=0.)
            self.tachometer = pulse_counter.FrequencyCounter(self.printer, tach_pin, sample_time, poll_time)

        self._setup_queue(self.mcu_dir_pin)
        self._setup_queue(self.mcu_pwm_pin)

        self.printer.register_event_handler('mmu:synced', self._handle_synced)
        self.printer.register_event_handler('mmu:unsynced', self._handle_unsynced)
        self.printer.register_event_handler('klippy:shutdown', self._handle_shutdown)

    def supports_gate(self, gate):
        return gate is not None and self.first_gate <= gate < self.first_gate + self.num_gates

    def _setup_queue(self, mcu_pin):
        mcu = mcu_pin.get_mcu()
        if mcu in self.gcrqs:
            return
        if hasattr(output_pin, 'GCodeRequestQueue'):
            self.gcrqs[mcu] = output_pin.GCodeRequestQueue(self.config, mcu, self._set_pin)
        else:
            self.gcrqs[mcu] = FallbackGCodeRequestQueue(self.config, mcu, self._set_pin)

    def _set_pin(self, print_time, req):
        mcu_pin, value = req
        if mcu_pin is self.mcu_pwm_pin:
            if value == self.last_pwm:
                return 'discard', 0.
            self.last_pwm = value
            self.mcu_pwm_pin.set_pwm(print_time, value)
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
        if self.last_dir is None or self.last_dir == desired_dir:
            self._send_pin(self.mcu_dir_pin, desired_dir, print_time)
            return

        if self.last_pwm > 0.:
            base = print_time or self._get_print_time(self.mcu_pwm_pin.get_mcu())
            self._send_pin(self.mcu_pwm_pin, 0., base)
            self._send_pin(self.mcu_dir_pin, desired_dir, base + self.dir_change_deadtime)
        else:
            self._send_pin(self.mcu_dir_pin, desired_dir, print_time)

    def _get_print_time(self, mcu):
        systime = self.reactor.monotonic()
        return mcu.estimated_print_time(systime + mcu.min_schedule_time())

    def _rpm_to_pwm(self, rpm):
        if rpm <= 0.:
            return 0.
        ratio = min(rpm / self.max_rpm, 1.)
        return self.pwm_min + (self.pwm_max - self.pwm_min) * ratio

    def _set_target(self, rpm, forward):
        rpm = max(0., min(rpm, self.max_rpm))
        pwm = self._rpm_to_pwm(rpm)
        t0 = self._get_print_time(self.mcu_pwm_pin.get_mcu())
        self._safe_set_direction(forward, t0)
        if self.last_dir != (1 if forward else 0):
            t0 += self.dir_change_deadtime
        self._send_pin(self.mcu_pwm_pin, pwm, t0)

    def stop(self):
        self._send_pin(self.mcu_pwm_pin, 0.)

    def run_distance(self, dist, speed):
        if dist == 0.:
            return 0.
        speed = abs(speed)
        if speed <= 0.:
            return 0.

        forward = dist > 0.
        rpm = 60. * speed / self.rotation_distance
        self._set_target(rpm, forward)
        self.reactor.pause(self.reactor.monotonic() + abs(dist) / speed)
        self.stop()
        return dist

    def start_move(self, dist, speed):
        if dist == 0.:
            self.stop()
            return
        speed = abs(speed)
        if speed <= 0.:
            self.stop()
            return
        forward = dist > 0.
        rpm = 60. * speed / self.rotation_distance
        self._set_target(rpm, forward)

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

    def get_rpm(self):
        if self.tachometer is None:
            return None
        return self.tachometer.get_frequency() * 30. / self.tach_ppr

    def get_status(self, _eventtime):
        return {
            'active': self.last_pwm > 0.,
            'pwm': self.last_pwm,
            'dir': self.last_dir,
            'rpm': self.get_rpm(),
            'rotation_distance': self.rotation_distance,
            'gate_start': self.first_gate,
            'gate_end': self.first_gate + self.num_gates - 1,
        }

    def _handle_synced(self):
        self.sync_active = True
        self.last_sync_eventtime = None
        self.last_sync_extruder_pos = None
        if self.sync_timer is None:
            self.sync_timer = self.reactor.register_timer(self._sync_update)
        self.reactor.update_timer(self.sync_timer, self.reactor.NOW)

    def _handle_unsynced(self):
        self.sync_active = False
        self.last_sync_eventtime = None
        self.last_sync_extruder_pos = None
        self.manual_hold_active = False
        self.manual_hold_until = None
        self.stop()
        if self.sync_timer is not None:
            self.reactor.update_timer(self.sync_timer, self.reactor.NEVER)

    def _handle_shutdown(self):
        self.stop()

    def _sync_update(self, eventtime):
        if not self.sync_active:
            return self.reactor.NEVER

        if self.manual_hold_active:
            if self.manual_hold_until is None or eventtime < self.manual_hold_until:
                return eventtime + self.SYNC_POLL_INTERVAL
            self.manual_hold_active = False
            self.manual_hold_until = None

        pos = self.mmu.toolhead.get_position()[3]
        if self.last_sync_eventtime is None:
            self.last_sync_eventtime = eventtime
            self.last_sync_extruder_pos = pos
            return eventtime + self.SYNC_POLL_INTERVAL

        dt = eventtime - self.last_sync_eventtime
        de = pos - self.last_sync_extruder_pos
        self.last_sync_eventtime = eventtime
        self.last_sync_extruder_pos = pos

        if dt <= 0.:
            return eventtime + self.SYNC_POLL_INTERVAL

        speed = de / dt
        if abs(speed) <= self.sync_min_speed:
            self.stop()
            return eventtime + self.SYNC_POLL_INTERVAL

        rpm = 60. * abs(speed) / self.rotation_distance
        self._set_target(rpm, speed >= 0.)

        if self.tachometer is not None:
            measured = self.get_rpm()
            if measured is not None and measured > (self.max_rpm * 1.15):
                self.stop()
                self.sync_active = False
                self.mmu.log_error('BLDC overspeed detected during sync; motor stopped')
                return self.reactor.NEVER

        return eventtime + self.SYNC_POLL_INTERVAL
