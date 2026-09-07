# Happy Hare MMU Software
#
# Copyright (C) 2022-2026  moggieuk#6538 (discord)
#                          moggieuk@hotmail.com
#
# Goal: Verify native per-unit BLDC construction, ownership and motion contracts.
#
#
# (\_/)
# ( *,*)
# (")_(") Happy Hare Ready
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#

"""Native v4 BLDC construction, ownership and drive contracts."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from test.hh import session
from test.hh.profiles import BOXTURTLE, Profile, UnitProfile


class TestBldcV4(unittest.TestCase):
    BLDC_UNIT_INDEX = 0

    @classmethod
    def profile(cls):
        return BOXTURTLE.derive('bldc', syms={
            'MMU_HAS_BLDC': True,
            'PIN_BLDC_PWM': 'unit0:BLDC_PWM',
            'PIN_BLDC_DIR': 'unit0:BLDC_DIR',
            'PIN_BLDC_TACHOMETER': 'unit0:BLDC_TACH',
            'PARAM_BLDC_ROTATION_DISTANCE': '2.0',
        })

    @classmethod
    def setUpClass(cls):
        cls.hh = session(cls.profile())
        cls.hh.boot(calibrate=True, selected_gate=cls.BLDC_UNIT_INDEX * 4)
        cls.drive = cls.hh.mmu.mmu_machine.units[cls.BLDC_UNIT_INDEX].drives[0]

    @classmethod
    def tearDownClass(cls):
        cls.hh.close()

    def test_constructs_without_gear_stepper(self):
        """A BLDC unit must not allocate fake MCU stepper pins."""
        self.assertIsNone(self.drive.mmu_gear_stepper)

    def test_status_is_readable(self):
        """The migrated tachometer status must not raise KeyError."""
        self.assertIn('tachometer_frequency', self.drive.get_status(0))

    def test_sensor_rail_has_no_stepper(self):
        """Sensor registration must not bind a nonexistent MCU stepper."""
        self.assertEqual(self.drive.rail.get_steppers(), [])

    def test_global_sync_event_does_not_start_bldc(self):
        """Only its own drive may enable the BLDC in mixed topologies."""
        self.hh.printer.send_event('mmu:synced')
        self.assertFalse(self.drive.bldc.sync_active)

    def test_espooler_shares_mcu_queue(self):
        """Both motor owners must dispatch through one MCU queue."""
        spooler = self.drive.mmu_unit.espooler
        self.assertIs(spooler.gcrqs[self.drive.bldc.mcu],
                      self.drive.bldc.pin_request_queue)

    def test_rotation_distance_uses_drive(self):
        """v4 calibration and sync feedback must reach the motor controller."""
        self.drive.set_rotation_distance(3.)
        self.assertEqual(self.drive.bldc.get_rotation_distance(), 3.)
        self.drive.set_rotation_distance(2.)

    def test_no_construction_errors(self):
        """A successful config load must not hide logged integration errors."""
        self.assertEqual(self.hh.errors, [])

    def test_drive_owns_all_sync_transitions(self):
        """The four modes must enable BLDC following only for extruder-led motion."""
        from extras.mmu.mmu_constants import (
            DRIVE_GEAR_SYNCED_TO_EXTRUDER, DRIVE_EXTRUDER_SYNCED_TO_GEAR,
            DRIVE_EXTRUDER_ONLY, DRIVE_UNSYNCED)
        states = []
        try:
            for mode in (DRIVE_GEAR_SYNCED_TO_EXTRUDER, DRIVE_EXTRUDER_SYNCED_TO_GEAR,
                         DRIVE_EXTRUDER_ONLY, DRIVE_UNSYNCED):
                self.drive.sync_mode(mode)
                states.append(self.drive.bldc.sync_active)
        finally:
            self.drive.sync_mode(DRIVE_UNSYNCED)
        self.assertEqual(states, [True, False, False, False])

    def test_saved_calibration_is_namespaced(self):
        """Calibration of one BLDC must not replace another unit's persisted map."""
        payload = {'points': [{'pwm': 0.3, 'rpm': 1000.}, {'pwm': 1., 'rpm': 5000.}]}
        manager = self.drive.mmu_machine.var_manager
        with patch.object(self.drive.bldc, 'get_calibration_map_payload', return_value=payload):
            self.drive.save_calibration()
        self.assertEqual(manager.get(self.drive.CALIBRATION_VARIABLE, None,
                                     namespace=self.drive.mmu_unit.name), payload)


class TestBldcMixed(TestBldcV4):
    BLDC_UNIT_INDEX = 1

    @classmethod
    def profile(cls):
        stepper = UnitProfile('unit0', BOXTURTLE.syms, index=0)
        bldc = UnitProfile('unit1', dict(BOXTURTLE.syms, **{
            'MMU_HAS_BLDC': True,
            'PIN_BLDC_PWM': 'unit1:BLDC_PWM',
            'PIN_BLDC_DIR': 'unit1:BLDC_DIR',
            'PIN_BLDC_TACHOMETER': 'unit1:BLDC_TACH',
            'PARAM_BLDC_ROTATION_DISTANCE': '2.0',
        }), index=1)
        return Profile('mixed_bldc', units=[stepper, bldc])

    def test_stepper_unit_keeps_native_drive(self):
        """BLDC configuration must not replace another unit's stepper drive."""
        stepper_drive = self.hh.mmu.mmu_machine.units[0].drives[0]
        self.assertIsNotNone(stepper_drive.mmu_gear_stepper)

    def test_gate_direction_uses_unit_local_index(self):
        """A nonzero first_gate must still select the correct direction map entry."""
        self.assertEqual(self.drive.bldc._map_distance_for_gate(10., gate=5)[0], -10.)


class TestBldcDriveMotion(unittest.TestCase):
    def setUp(self):
        from test.hh.root import install
        install()
        from extras.mmu.unit.mmu_bldc_drive import MmuBldcDrive
        from extras.mmu.mmu_constants import DRIVE_UNSYNCED
        self.drive = object.__new__(MmuBldcDrive)
        self.drive._sync_mode = DRIVE_UNSYNCED
        self.drive._position = 0.
        self.drive.printer = SimpleNamespace(command_error=RuntimeError)

    def test_finite_move_reports_tach_distance(self):
        """Position must describe measured travel, including overshoot."""
        bldc = SimpleNamespace(
            tachometer=SimpleNamespace(get_count_sample=lambda: (0., 0.)),
            start_move=lambda dist, speed: 1., wait_for_move=lambda wait: None,
            _finite_move_tracker=SimpleNamespace(aborted=False, completed=True),
            stop=lambda: None)
        self.drive.bldc = bldc
        with patch.object(self.drive, '_travelled', return_value=12.):
            self.assertEqual(self.drive.move(10., 20., 100.), (12., False))

    def test_failed_finite_move_stops_and_raises(self):
        """A tach stall cannot be reported as a successful full move."""
        stops = []
        self.drive.bldc = SimpleNamespace(
            tachometer=SimpleNamespace(get_count_sample=lambda: (0., 0.)),
            start_move=lambda dist, speed: 1., wait_for_move=lambda wait: None,
            _finite_move_tracker=SimpleNamespace(aborted=True, completed=False),
            stop=lambda: stops.append('stop'))
        with patch.object(self.drive, '_travelled', return_value=2.):
            with self.assertRaisesRegex(RuntimeError, 'tachometer target'):
                self.drive.move(10., 20., 100.)
        self.assertEqual((self.drive._position, stops), (2., ['stop']))

    def test_combined_move_submits_both_before_waiting(self):
        """Gear and extruder must be active concurrently, sharing the start time."""
        calls = []
        self.drive.bldc = SimpleNamespace(queue_trapzoid_move=lambda move, direction, pt, source:
                                         calls.append(('bldc', pt)))
        self.drive.mmu_extruder_stepper = SimpleNamespace(
            get_mode_position=lambda: 0.,
            do_move=lambda target, speed, accel: calls.append(('extruder', target)))
        self.drive.mmu = SimpleNamespace(movequeue_wait=lambda: calls.append(('wait',)))
        with patch.object(self.drive, '_prepare_extruder_move', return_value=20.):
            self.drive._move_with_extruder(10., 20., 100.)
        self.assertEqual(calls, [('bldc', 20.), ('extruder', 10.), ('wait',)])

    def test_homing_query_failure_stops_motor(self):
        """A sensor communication error must not leave an open motor command."""
        stops = []
        self.drive.bldc = SimpleNamespace(
            has_tachometer=lambda: True,
            tachometer=SimpleNamespace(get_count_sample=lambda: (0., 0.)),
            reactor=SimpleNamespace(monotonic=lambda: 0.),
            start_move=lambda dist, speed: 1., POSITION_STANDSTILL_TIMEOUT_S=1.,
            _get_scheduled_print_time=lambda: 0.)
        queries = iter([False, RuntimeError('sensor disconnected')])
        def fail_query(print_time):
            result = next(queries)
            if isinstance(result, Exception):
                raise result
            return result
        endstop = SimpleNamespace(query_endstop=fail_query)
        with patch.object(self.drive, '_brake_and_wait', side_effect=lambda: stops.append('stop')), \
                patch.object(self.drive, '_travelled', return_value=2.):
            with self.assertRaisesRegex(RuntimeError, 'sensor disconnected'):
                self.drive._home(10., 20., endstop, True)
        self.assertEqual((self.drive._position, stops), (2., ['stop']))

    def test_brake_wait_error_still_stops_motor(self):
        """Reactor shutdown during braking must still issue the stop path."""
        stops = []
        def fail_pause(waketime):
            raise RuntimeError('reactor stopped')
        self.drive.bldc = SimpleNamespace(
            brake_to_stop=lambda: None, brake_max_time=0.25, min_schedule_time=0.25,
            tachometer=SimpleNamespace(tachometer_sample_time=0.04),
            reactor=SimpleNamespace(monotonic=lambda: 0., pause=fail_pause),
            stop=lambda: stops.append('stop'))
        with self.assertRaisesRegex(RuntimeError, 'reactor stopped'):
            self.drive._brake_and_wait()
        self.assertEqual(stops, ['stop'])

    def test_already_triggered_endstop_does_not_start_motor(self):
        """An already-satisfied home must not kick the motor into the endstop."""
        self.drive.bldc = SimpleNamespace(
            has_tachometer=lambda: True,
            tachometer=SimpleNamespace(get_count_sample=lambda: (0., 0.)),
            reactor=SimpleNamespace(), _get_scheduled_print_time=lambda: 0.)
        endstop = SimpleNamespace(query_endstop=lambda pt: True)
        self.assertEqual(self.drive._home(10., 20., endstop, True), (0., True))

    def test_legacy_calibration_loads_by_original_unit_index(self):
        """Existing v3 maps remain readable before saving in the v4 namespace."""
        payload = {'points': [{'pwm': 0.3, 'rpm': 1000.}, {'pwm': 1., 'rpm': 5000.}]}
        loaded = []
        self.drive.mmu_unit = SimpleNamespace(name='second', unit_index=1)
        self.drive.mmu_machine = SimpleNamespace(
            mmu_controller=SimpleNamespace(log_info=lambda msg: None),
            var_manager=SimpleNamespace(get=lambda key, default, **kwargs:
                                        {'unit_1': payload} if key == 'mmu_bldc_map' else None))
        self.drive.bldc = SimpleNamespace(set_calibration_map=loaded.append)
        self.drive.handle_connect()
        self.assertEqual(loaded, [payload])

    def test_reverse_compound_home_records_matching_child(self):
        """Release homing must preserve the identity of the child that released."""
        from extras.mmu.mmu_sensor_utils import MmuCompoundEndstop
        class Endstop:
            def __init__(self, state):
                self.state = state
            def query_endstop(self, print_time):
                return self.state
        compound = MmuCompoundEndstop(self.drive.printer, 'both',
                                      [(Endstop(True), 'first'), (Endstop(False), 'second')])
        self.assertEqual((compound.query_homing_endstop(1., False),
                          compound.get_triggered_endstop_name()), (True, 'second'))

    def test_fallback_queue_dispatches_multiple_owners(self):
        """Kalico's fallback must retain shared queue ownership and callback routing."""
        from extras.mmu.mmu_pin_queue import MmuPinRequestQueue
        from extras import output_pin
        calls = []
        printer = SimpleNamespace(register_event_handler=lambda event, cb: None)
        config = SimpleNamespace(get_printer=lambda: printer)
        # MCU objects are identity keys in production.
        class Mcu:
            def register_flush_callback(self, callback):
                pass
        mcu = Mcu()
        queue_class = output_pin.GCodeRequestQueue
        del output_pin.GCodeRequestQueue
        try:
            first = MmuPinRequestQueue.get_for_mcu(config, mcu)
            second = MmuPinRequestQueue.get_for_mcu(config, mcu)
            first.send((lambda pt, value: calls.append(('bldc', value)), 0.5), 1.)
            second.send((lambda pt, value: calls.append(('spooler', value)), 0.7), 2.)
        finally:
            output_pin.GCodeRequestQueue = queue_class
        self.assertEqual((first is second, calls),
                         (True, [('bldc', 0.5), ('spooler', 0.7)]))


class TestYammuConfig(unittest.TestCase):
    def test_installer_renders_bldc_and_local_servo_macro(self):
        """A YAMMU config must select its own gates without allocating gear steppers."""
        from test.hh import cfg
        profile = Profile('yammu', syms={
            'MMU_TYPE_YAMMU_1_0': True,
            'PIN_BLDC_PWM': 'unit0:BLDC_PWM',
            'PIN_BLDC_DIR': 'unit0:BLDC_DIR',
            'PIN_BLDC_TACHOMETER': 'unit0:BLDC_TACH',
            'PARAM_YAMMU_SERVO_PINS': 'unit0:SERVO0, unit0:SERVO1',
        })
        rendered = cfg.render(profile)
        config = cfg.assemble(rendered)
        self.assertEqual((
            config.has_section('mmu_gear_bldc unit0'),
            config.has_section('mmu_stepper unit0_gear'),
            config.get('mmu_unit_parameters unit0', 'select_tool_macro'),
            config.has_section('servo unit0_yammu_1'),
            'params.LGATE' in config.get('gcode_macro _YAMMU_SELECT_TOOL_unit0', 'gcode'),
        ), (True, False, '_YAMMU_SELECT_TOOL_unit0', True, True))
