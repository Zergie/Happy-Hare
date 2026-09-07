# Happy Hare MMU Software
#
# Copyright (C) 2022-2026  moggieuk#6538 (discord)
#                          moggieuk@hotmail.com
#
# Goal: Adapt BLDC motion to the per-unit drive and sensor interfaces.
#
#
# (\_/)
# ( *,*)
# (")_(") Happy Hare Ready
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#

import math
from types import SimpleNamespace

from ... import force_move
from ...mmu_stepper import MmuGenericRail
from ..mmu_constants import (
    DRIVE_UNSYNCED, DRIVE_EXTRUDER_ONLY, DRIVE_EXTRUDER_SYNCED_TO_GEAR,
    DRIVE_GEAR_SYNCED_TO_EXTRUDER, DRIVE_MODE_NAMES,
)
from ..mmu_gear_bldc import MmuGearBldc
from .mmu_drive import MmuDrive


class MmuBldcDrive(MmuDrive):
    """A per-unit BLDC drive with native v4 sensor and extruder ownership."""

    HOMING_POLL_TIME = 0.01
    CALIBRATION_VARIABLE = 'mmu_bldc_calibration_map'

    def __init__(self, config, mmu_unit):
        self.printer = config.get_printer()
        self.name = config.get_name()
        self.mmu_unit = mmu_unit
        self.mmu_machine = mmu_unit.mmu_machine
        self.mmu_extruder_stepper = mmu_unit.extruder_wrapper.homing_extruder_stepper
        self.mmu_gear_stepper = None
        self._sync_mode = DRIVE_UNSYNCED
        self._position = 0.
        self._tmc = self._default_current = None
        self._run_current_percent = 100
        self.bldc = MmuGearBldc(config, None, mmu_unit.first_gate, mmu_unit.num_gates)
        self.bldc.mmu_unit = mmu_unit
        self._rail = MmuGenericRail(None, config, need_position_minmax=False,
                                    default_position_endstop=0.)
        self.printer.add_object(config.get_name(), self.bldc)
        self.printer.register_event_handler('klippy:connect', self.handle_connect)

    @property
    def rail(self):
        return self._rail

    def handle_connect(self):
        self.mmu = self.mmu_machine.mmu_controller
        payload = self.mmu_machine.var_manager.get(
            self.CALIBRATION_VARIABLE, None, namespace=self.mmu_unit.name)
        if payload is None:
            legacy = self.mmu_machine.var_manager.get('mmu_bldc_map', {})
            if isinstance(legacy, dict):
                payload = legacy.get('unit_%d' % self.mmu_unit.unit_index)
                if payload is not None:
                    self.mmu.log_info('Loaded legacy BLDC map for unit %s; preserve unit order when upgrading'
                                      % self.mmu_unit.name)
        if payload is not None:
            self.bldc.set_calibration_map(payload)

    def save_calibration(self):
        self.mmu_machine.var_manager.set(
            self.CALIBRATION_VARIABLE, self.bldc.get_calibration_map_payload(),
            namespace=self.mmu_unit.name, write=True)

    def sync_mode(self, mode):
        if mode not in DRIVE_MODE_NAMES:
            raise self.printer.command_error("Invalid MMU drive sync mode: %s" % mode)
        previous = self._sync_mode
        if previous == mode:
            if mode == DRIVE_GEAR_SYNCED_TO_EXTRUDER and not self.bldc.sync_active:
                self.bldc.set_sync_enabled(True)
                return True
            return False
        self.mmu.movequeue_wait()
        position = self.get_filament_position()
        self.bldc.set_sync_enabled(False)
        extruder = self.mmu_extruder_stepper
        if mode in (DRIVE_EXTRUDER_ONLY, DRIVE_EXTRUDER_SYNCED_TO_GEAR):
            extruder.switch_to_manual_mode()
        else:
            extruder.switch_to_extruder_mode()
        extruder.do_set_position(position)
        self._position = position
        self._sync_mode = mode
        if mode == DRIVE_GEAR_SYNCED_TO_EXTRUDER:
            self.bldc.set_sync_enabled(True)
            self.printer.send_event('mmu:synced')
        elif previous == DRIVE_GEAR_SYNCED_TO_EXTRUDER:
            self.printer.send_event('mmu:unsynced')
        return True

    def enable_motor(self, on):
        if not on:
            self.bldc.set_sync_enabled(False)

    def set_filament_position(self, position):
        self._position = position
        if self._sync_mode != DRIVE_UNSYNCED:
            self.mmu_extruder_stepper.do_set_position(position)

    def get_filament_position(self):
        if self._sync_mode != DRIVE_UNSYNCED:
            return self.mmu_extruder_stepper.get_mode_position()
        return self._position

    def get_live_filament_position(self):
        if self._sync_mode != DRIVE_UNSYNCED:
            stepper = self.mmu_extruder_stepper.stepper
            return stepper.get_mcu_position() * stepper.get_step_dist()
        return self._position

    def driving_stepper(self):
        return self.mmu_extruder_stepper if self._sync_mode != DRIVE_UNSYNCED else None

    def _homing_rail(self):
        if self._sync_mode == DRIVE_EXTRUDER_ONLY:
            return self.mmu_extruder_stepper.rail
        return self.rail

    def has_endstop(self, name):
        return self._homing_rail().has_endstop(name)

    def get_endstop(self, name):
        return self._homing_rail().get_homing_endstops(name)[0][0]

    def get_extra_endstop_names(self):
        return self._homing_rail().get_extra_endstop_names()

    def is_endstop_virtual(self, name):
        return self._homing_rail().is_endstop_virtual(name)

    def set_gear_direction(self, direction):
        gate = self.mmu_unit.local_gate(self.mmu.gate_selected)
        self.bldc.direction_map[gate] = int(bool(direction))

    def get_rotation_distance(self):
        return self.bldc.get_rotation_distance()

    def set_rotation_distance(self, distance):
        self.bldc.set_rotation_distance(distance)

    def calculate_rotation_distance(self, measured, requested):
        distance = self.bldc.calculate_rotation_distance(measured, requested)
        if distance is None:
            raise self.printer.command_error(
                "BLDC calibration requires a completed tachometer move of LENGTH")
        return distance

    def _travelled(self, start_count, requested):
        count, _ = self.bldc.tachometer.get_count_sample()
        if count is None or start_count is None:
            return requested
        revolutions = max(0., count - start_count) / self.bldc.tachometer.get_counts_per_revolution()
        return math.copysign(revolutions * self.get_rotation_distance(), requested)

    def _home(self, dist, speed, endstop, triggered):
        if not self.bldc.has_tachometer():
            raise self.printer.command_error("BLDC homing requires a tachometer")
        reactor = self.bldc.reactor
        start_count, _ = self.bldc.tachometer.get_count_sample()
        if start_count is None:
            raise self.printer.command_error("BLDC homing requires an initial tachometer sample")
        if self._endstop_triggered(endstop, triggered):
            return 0., True
        homed = False
        try:
            wait = self.bldc.start_move(dist, speed)
            deadline = reactor.monotonic() + wait + self.bldc.POSITION_STANDSTILL_TIMEOUT_S
            while reactor.monotonic() < deadline:
                homed = self._endstop_triggered(endstop, triggered)
                if homed or abs(self._travelled(start_count, dist)) >= abs(dist):
                    break
                tracker = self.bldc._finite_move_tracker
                if tracker is not None and (tracker.aborted or tracker.stop_dispatched):
                    break
                reactor.pause(reactor.monotonic() + self.HOMING_POLL_TIME)
        finally:
            self._brake_and_wait()
            self._position += self._travelled(start_count, dist)
        if not homed:
            raise self.printer.command_error("BLDC homing ended without triggering the endstop")
        return self._travelled(start_count, dist), True

    def _endstop_triggered(self, endstop, triggered):
        print_time = self.bldc._get_scheduled_print_time()
        query = getattr(endstop, 'query_homing_endstop', None)
        return (query(print_time, triggered) if query is not None
                else bool(endstop.query_endstop(print_time)) == triggered)

    def _brake_and_wait(self):
        try:
            self.bldc.brake_to_stop()
            reactor = self.bldc.reactor
            wait = (self.bldc.brake_max_time + self.bldc.min_schedule_time
                    + 2 * self.bldc.tachometer.tachometer_sample_time)
            reactor.pause(reactor.monotonic() + wait)
        finally:
            self.bldc.stop()

    def _move_with_extruder(self, dist, speed, accel):
        extruder = self.mmu_extruder_stepper
        print_time = self._prepare_extruder_move()
        direction, accel_t, cruise_t, cruise_v = force_move.calc_move_time(dist, speed, accel)
        profile = SimpleNamespace(accel_t=accel_t, cruise_t=cruise_t,
                                  decel_t=accel_t, accel=accel, start_v=0.,
                                  cruise_v=cruise_v, end_v=0.)
        self.bldc.queue_trapzoid_move(profile, direction, print_time, 'manual_extruder')
        extruder.do_move(extruder.get_mode_position() + dist, speed, accel)
        self.mmu.movequeue_wait()
        return dist, False

    def _prepare_extruder_move(self):
        extruder = self.mmu_extruder_stepper
        extruder.sync_print_time()
        earliest = self.bldc._get_scheduled_print_time() + self.bldc.min_schedule_time
        if extruder.next_cmd_time < earliest:
            self.mmu.toolhead.dwell(earliest - extruder.next_cmd_time)
            extruder.sync_print_time()
        return extruder.next_cmd_time

    def move(self, dist, speed, accel, homing_move=0, endstop_name='default'):
        if self._sync_mode == DRIVE_EXTRUDER_ONLY:
            self._driving_stepper = self.mmu_extruder_stepper
            return super().move(dist, speed, accel, homing_move, endstop_name)
        if self._sync_mode == DRIVE_GEAR_SYNCED_TO_EXTRUDER:
            raise self.printer.command_error("Synced BLDC moves must use the extruder toolhead")
        if self._sync_mode == DRIVE_EXTRUDER_SYNCED_TO_GEAR:
            speed = min(speed, self.bldc.get_effective_max_linear_speed())
        if homing_move:
            if self.is_endstop_virtual(endstop_name):
                raise self.printer.command_error("BLDC drive cannot home using stepper StallGuard")
            if self._sync_mode == DRIVE_EXTRUDER_SYNCED_TO_GEAR:
                return self._home_with_extruder(dist, speed, accel, homing_move, endstop_name)
            return self._home(dist, speed, self.get_endstop(endstop_name), homing_move > 0)
        try:
            if self._sync_mode == DRIVE_EXTRUDER_SYNCED_TO_GEAR:
                return self._move_with_extruder(dist, speed, accel)
            count, _ = self.bldc.tachometer.get_count_sample()
            self.bldc.wait_for_move(self.bldc.start_move(dist, speed))
            actual = self._travelled(count, dist)
            self._position += actual
            tracker = self.bldc._finite_move_tracker
            if tracker is not None and (tracker.aborted or not tracker.completed):
                raise self.printer.command_error("BLDC move failed to reach its tachometer target")
            return actual, False
        except Exception:
            self.bldc.stop()
            raise

    def _home_with_extruder(self, dist, speed, accel, homing_move, endstop_name):
        extruder = self.mmu_extruder_stepper
        endstop = self.get_endstop(endstop_name)
        if self._endstop_triggered(endstop, homing_move > 0):
            return 0., True
        added_endstop = not extruder.rail.has_endstop(endstop_name)
        if added_endstop:
            extruder.rail.add_extra_endstop(None, endstop_name, register=False, mcu_endstop=endstop)
        self._driving_stepper = extruder
        try:
            self.bldc.start_move(None, math.copysign(speed, dist),
                                 print_time=self._prepare_extruder_move())
            return super().move(dist, speed, accel, homing_move, endstop_name)
        finally:
            try:
                self._brake_and_wait()
            finally:
                if added_endstop:
                    extruder.rail.remove_compound_endstop(endstop_name)

    def get_status(self, eventtime):
        return dict(self.bldc.get_status(eventtime),
                    sync_mode=self._sync_mode,
                    sync_mode_name=DRIVE_MODE_NAMES[self._sync_mode],
                    drive_stepper=None, filament_position=self.get_filament_position())
