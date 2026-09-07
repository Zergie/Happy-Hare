# Happy Hare MMU Software
#
# Copyright (C) 2022-2026  moggieuk#6538 (discord)
#                          moggieuk@hotmail.com
#
# Goal: Exercise BLDC control and calibration with deterministic fake hardware.
#
#
# (\_/)
# ( *,*)
# (")_(") Happy Hare Ready
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#
import os
import sys
import unittest
from unittest.mock import patch

from test.support.bldc_fakes import (
    FakeConfig,
    FakeExtruder,
    FakeMove,
    FakeMmu,
    FakePin,
    FakePrinter,
    FakeToolhead,
    StrictMcu,
    load_mmu_gear_bldc_module,
)


def _make_bldc_runtime(bldc_module):
    extruder = FakeExtruder()
    toolhead = FakeToolhead(extruder=extruder)
    printer = FakePrinter(toolhead=toolhead)
    mmu = FakeMmu(toolhead=toolhead, gate_selected=0)
    config = FakeConfig(
        printer,
        {
            "dir_pin": "D0",
            "pwm_pin": "P0",
            "pwm_min": 0.2,
            "pwm_max": 1.0,
            "rotation_distance": 2.0,
            "direction_map": [0],
            "kick_start_time": 0.05,
            "tachometer_pin": "T0",
        },
    )
    return bldc_module.MmuGearBldc(config, mmu)


def _make_mapped_gate_bldc(bldc_module):
    return _create_bldc(
        bldc_module,
        {
            "dir_pin": "D0",
            "pwm_pin": "P0",
            "rotation_distance": 2.0,
            "direction_map": [0, 1],
            "kick_start_time": 0.05,
            "tachometer_pin": "T0",
        },
        num_gates=2,
    )


def _create_bldc(bldc_module, config_values, num_gates=1):
    printer = FakePrinter(toolhead=FakeToolhead(extruder=FakeExtruder()))
    mmu = FakeMmu(toolhead=printer.lookup_object("toolhead"), gate_selected=0)
    config = FakeConfig(printer, config_values)
    return bldc_module.MmuGearBldc(config, mmu, num_gates=num_gates)


def advance_motion_through(bldc, dispatch_time):
    while bldc.reactor.monotonic() <= dispatch_time:
        wake_time = bldc.motion_timer["when"]
        if wake_time == bldc.reactor.NEVER:
            break
        bldc.reactor.pause(wake_time)


def direction_pin_values(bldc):
    return [value for _print_time, value in bldc.mcu_dir_pin.digital_calls]


def capture_pwm_requests(bldc):
    requests = []

    def capture(mcu_pin, value, print_time):
        if mcu_pin is bldc.mcu_pwm_pin:
            requests.append((print_time, value))

    def capture_batch(pin_updates, print_time):
        for mcu_pin, value in pin_updates:
            capture(mcu_pin, value, print_time)

    bldc._send_pin = capture
    bldc._send_pin_updates = capture_batch
    return requests


def set_test_calibration_map(bldc, minimum_rpm=1200.0):
    bldc.set_calibration_map(
        {
            "pwm_min": 0.3,
            "points": [
                {"pwm": 0.3, "rpm": minimum_rpm},
                {"pwm": 0.6, "rpm": minimum_rpm * 2.0},
                {"pwm": 1.0, "rpm": minimum_rpm * 3.0},
            ],
        }
    )


def _create_sync_monitor(bldc_module, min_retract_distance):
    bldc = _create_bldc(
        bldc_module,
        {
            "dir_pin": "D0",
            "pwm_pin": "P0",
            "sync_retract_min_distance": min_retract_distance,
        },
    )
    bldc.sync_active = True
    monitor = bldc_module.ProcessMoveSyncMonitor(bldc.mmu)
    monitor.active_bldc = bldc
    return bldc, monitor


def _create_extruder_move(distance):
    axis_ratio = 1.0 if distance > 0.0 else -1.0
    move = FakeMove([0.0, 0.0, 0.0, axis_ratio], cruise_v=30.0)
    move.move_d = abs(distance)
    return move


def prepare_fresh_tachometer(bldc):
    bldc.tachometer.last_tach_eventtime = bldc._get_scheduled_print_time() + bldc.min_schedule_time


def active_print_time(bldc):
    return bldc._get_scheduled_print_time() + bldc.min_schedule_time


def prime_tachometer_count(bldc, count=0.0):
    bldc.tachometer.handle_tachometer(0.0, count, 0.0)


def lock_tracker_for_settling(bldc):
    bldc.set_rotation_distance(7.5)
    prime_tachometer_count(bldc)
    bldc.start_move(75.0, 600.0)
    tracker = bldc._finite_move_tracker
    bldc._stop_motion_timer()
    tracker.actual_stop_print_time = tracker.start_print_time + 0.1
    tracker.stop_descriptor.print_time = tracker.actual_stop_print_time
    tracker.endpoint_locked = True
    tracker.stop_dispatched = True
    tracker.completed = True
    return tracker


def settle_tracker_count(bldc, tracker, final_count, stop_time=None):
    bldc._stop_motion_timer()
    stop_time = stop_time or tracker.stop_descriptor.print_time
    tracker.stop_descriptor.print_time = stop_time
    tracker.actual_stop_print_time = stop_time
    tracker.endpoint_locked = True
    tracker.stop_dispatched = True
    tracker.completed = True
    sample_time = stop_time
    for count in (final_count, final_count, final_count):
        sample_time += bldc.tachometer.tachometer_sample_time
        bldc.reactor.pause(sample_time)
        bldc.tachometer.handle_tachometer(sample_time, count, sample_time)


class TestBldcStandaloneImport(unittest.TestCase):
    def test_loads_without_external_klipper_checkout(self):
        """A clean CI checkout must supply every BLDC test dependency."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        with patch.dict(sys.modules), patch("os.path.exists", return_value=False):
            module = load_mmu_gear_bldc_module(repo_root)
            bldc = _make_bldc_runtime(module)
            bldc.start_move(100.0, 20.0)
            self.assertTrue(bldc.motion_queue)


class TestMmuGearBldc(unittest.TestCase):
    def setUp(self):
        modules = patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)
        repo_root = os.path.dirname(os.path.dirname(__file__))
        self.bldc_module = load_mmu_gear_bldc_module(repo_root)

    def assertApproxEqual(self, actual, expected, abs_tol=None):
        """Preserve the controller checks' relative and absolute tolerances."""
        if isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for index, (value, target) in enumerate(zip(actual, expected)):
                with self.subTest(index=index):
                    self.assertApproxEqual(value, target, abs_tol)
            return
        if not isinstance(expected, (int, float)):
            self.assertEqual(actual, expected)
            return
        tolerance = max(1e-12, 1e-6 * abs(expected)) if abs_tol is None else abs_tol
        self.assertAlmostEqual(actual, expected, delta=tolerance)

    def test_configuration_defaults_match_yammu_hardware(self):
        """Runtime defaults must stay aligned with shipped YAMMU configuration."""
        bldc_module = self.bldc_module
        bldc = _create_bldc(bldc_module, {"dir_pin": "D0", "pwm_pin": "P0"}, num_gates=4)
        self.assertApproxEqual(bldc.pwm_min, 0.2)
        self.assertApproxEqual(bldc.torque_assist_pwm, 0.2)
        self.assertApproxEqual(bldc.sync_retract_min_distance, 1.0)
        self.assertApproxEqual(bldc.tachometer.tachometer_sample_time, 0.04)
        self.assertApproxEqual(bldc.tachometer.tachometer_stale_time, 0.12)
        self.assertApproxEqual(bldc.tachometer.tachometer_poll_interval, 0.00025)
        self.assertEqual(bldc.direction_map, [0, 1, 0, 1])

    def test_configuration_rejects_torque_assist_above_pwm_max(self):
        """Torque-assist power must never bypass configured motor power ceiling."""
        bldc_module = self.bldc_module
        with self.assertRaisesRegex(RuntimeError, "torque_assist_pwm"):
            _create_bldc(
                bldc_module,
                {"dir_pin": "D0", "pwm_pin": "P0", "pwm_max": 0.4, "torque_assist_pwm": 0.5},
            )

    def test_zero_torque_assist_pwm_preserves_legacy_speed_control(self):
        """Zero power provides an explicit compatibility switch for torque assist."""
        bldc_module = self.bldc_module
        bldc = _create_bldc(
            bldc_module,
            {
                "dir_pin": "D0",
                "pwm_pin": "P0",
                "tachometer_pin": "T0",
                "rotation_distance": 2.0,
                "torque_assist_pwm": 0.0,
            },
        )
        set_test_calibration_map(bldc)
        bldc.start_move(None, 5.0)
        self.assertTrue(
            not any(
                (
                    isinstance(descriptor, bldc_module.MotionTorqueAssist)
                    for descriptor, _source in bldc.motion_queue
                )
            )
        )

    def test_tachometer_clamps_poll_interval_before_edges_can_alias(self):
        """Polling must sample fast enough to count both tach edges at max RPM."""
        bldc_module = self.bldc_module
        bldc = _create_bldc(
            bldc_module,
            {
                "dir_pin": "D0",
                "pwm_pin": "P0",
                "tachometer_pin": "T0",
                "tachometer_ppr": 9,
                "tachometer_poll_interval": 0.001,
            },
        )
        self.assertApproxEqual(
            bldc.tachometer.tachometer.poll_interval, 60.0 / (5500.0 * 18.0 * 2.0)
        )
        self.assertIn("clamped", bldc.mmu.warning_logs[-1])

    def test_tachometer_preserves_safe_poll_interval(self):
        """A faster user poll interval must remain unchanged."""
        bldc_module = self.bldc_module
        bldc = _create_bldc(
            bldc_module,
            {
                "dir_pin": "D0",
                "pwm_pin": "P0",
                "tachometer_pin": "T0",
                "tachometer_ppr": 9,
                "tachometer_poll_interval": 0.0002,
            },
        )
        self.assertApproxEqual(bldc.tachometer.tachometer.poll_interval, 0.0002)
        self.assertEqual(bldc.mmu.warning_logs, [])

    def test_calibrated_max_rpm_cannot_exceed_tach_polling_limit(self):
        """RPM control must never enter a range where GPIO polling aliases edges."""
        bldc_module = self.bldc_module
        bldc = _create_bldc(
            bldc_module,
            {
                "dir_pin": "D0",
                "pwm_pin": "P0",
                "tachometer_pin": "T0",
                "tachometer_ppr": 9,
                "tachometer_poll_interval": 0.0003,
            },
        )
        bldc.calibration_map_points = [{"pwm": 1.0, "rpm": 8000.0}]
        self.assertApproxEqual(bldc.get_effective_max_rpm(), 60.0 / (0.0003 * 18.0 * 2.0))

    def test_position_feed_forward_normalizes_frequency_across_commanded_rpm(self):
        """Learning at lower RPM must not corrupt prediction for a later faster move."""
        bldc_module = self.bldc_module
        feed_forward = bldc_module.PositionFeedForwardState(1.0, 0.5)
        feed_forward.learn_frequency(100.0, 4000.0)
        feed_forward.learn_frequency(50.0, 2000.0)
        self.assertApproxEqual(feed_forward.estimate_frequency(4000.0), 100.0)

    def test_position_feed_forward_learns_from_applied_capped_stop_lead(self):
        """Short-move lead cap must remain basis for settled position correction."""
        bldc_module = self.bldc_module
        feed_forward = bldc_module.PositionFeedForwardState(10.0, 0.5)
        feed_forward.learn_stop_lead(2.0, -1.0, 5.0)
        self.assertApproxEqual(feed_forward.stop_lead_counts, 1.5)

    def test_pin_updates_reserve_transmission_lead_for_delayed_mcu_delivery(self):
        """Queued PWM must remain valid after a delayed motion-queue flush."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        mcu = StrictMcu(
            min_schedule=0.1, base_print_time=5.0, reactor=bldc.reactor, delivery_latency=0.25
        )
        bldc.mcu = mcu
        bldc.mcu_dir_pin = FakePin("D0", mcu)
        bldc.mcu_pwm_pin = FakePin("P0", mcu)
        bldc.min_schedule_time = mcu.min_schedule_time() + bldc.PIN_SCHEDULE_LEAD_S
        bldc._get_scheduled_print_time = lambda: 5.0
        bldc._send_pin_updates(((bldc.mcu_pwm_pin, 0.5),), 5.0)
        self.assertEqual(bldc.mcu_pwm_pin.pwm_calls, [(5.35, 0.5)])

    def test_short_finite_move_keeps_one_tach_sample_before_commit(self):
        """Keep one tach interval available to correct a short 100 mm move."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc.set_rotation_distance(12.6846)
        prime_tachometer_count(bldc)
        bldc.start_move(100.0, 600.0)
        tracker = bldc._finite_move_tracker
        adjustable_time = (
            tracker.stop_descriptor.print_time
            - tracker.start_print_time
            - bldc.mcu.min_schedule_time()
        )
        self.assertGreaterEqual(adjustable_time, bldc.tachometer.tachometer_sample_time)

    def test_short_finite_move_skips_kick_that_cannot_be_queued(self):
        """Skip kick when queue spacing would delay terminal PWM zero."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc = bldc_runtime
        bldc.mcu._min_schedule = 0.1
        bldc.min_schedule_time = 0.11
        bldc.direction_setup_time = 0.1
        bldc.motion_sample_time = 0.04
        bldc.last_dir = 1
        bldc.set_rotation_distance(12.6846)
        prime_tachometer_count(bldc)
        bldc.start_move(100.0, 600.0)
        direct_pwm_moves = [
            descriptor
            for descriptor, _source in bldc.motion_queue
            if isinstance(descriptor, bldc_module.MotionPwmDirect)
        ]
        self.assertEqual(direct_pwm_moves, [])

    def test_short_hardware_move_dispatches_pwm_stop_at_target(self):
        """Hardware queue spacing must not delay terminal PWM zero."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc.mcu._min_schedule = 0.1
        bldc.min_schedule_time = 0.11
        bldc.direction_setup_time = 0.1
        bldc.motion_sample_time = 0.04
        bldc.last_dir = 1
        bldc.set_rotation_distance(12.6846)
        prime_tachometer_count(bldc)
        bldc.start_move(100.0, 600.0)
        tracker = bldc._finite_move_tracker
        target_stop_time = tracker.stop_descriptor.print_time
        for _iteration in range(10):
            wake_time = bldc.motion_timer["when"]
            if wake_time == bldc.reactor.NEVER:
                break
            bldc.reactor.pause(wake_time)
        pwm_calls = bldc.mcu_pwm_pin.pwm_calls
        self.assertApproxEqual(
            (len(pwm_calls), pwm_calls[0][0], pwm_calls[-1][0], pwm_calls[-1][1]),
            (2, tracker.start_print_time, target_stop_time, 0.0),
        )

    def test_hardware_trace_uses_live_coast_lead_without_blocking_stop(self):
        """Replay real first samples and preserve queue space for early PWM zero."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc.mcu._min_schedule = 0.1
        bldc.min_schedule_time = 0.11
        bldc.direction_setup_time = 0.1
        bldc.motion_sample_time = 0.04
        bldc.last_dir = 1
        bldc.set_rotation_distance(12.6846)
        prime_tachometer_count(bldc)
        bldc.start_move(100.0, 600.0)
        tracker = bldc._finite_move_tracker
        for sample_offset, count in ((0.043187, 22.0), (0.084187, 53.0)):
            sample_time = tracker.start_print_time + sample_offset
            bldc.reactor.pause(sample_time)
            bldc.tachometer.handle_tachometer(sample_time, count, sample_time)
        pwm_calls = bldc.mcu_pwm_pin.pwm_calls
        self.assertApproxEqual(
            (
                tracker.stop_lead_counts,
                len(pwm_calls),
                pwm_calls[-1][1],
                pwm_calls[-1][0] - tracker.start_print_time,
            ),
            (25.3103, 2, 0.0, 0.194187),
            abs_tol=0.0001,
        )

    def test_nonzero_post_stop_sample_does_not_finalize_learning(self):
        """Do not mistake a coasting sample for the settled endpoint."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        tracker = lock_tracker_for_settling(bldc)
        sample_time = tracker.actual_stop_print_time + 0.04
        bldc.reactor.pause(sample_time)
        bldc.tachometer.handle_tachometer(sample_time, 20.0, sample_time)
        self.assertIs(tracker.learning_completed, False)

    def test_two_zero_samples_freeze_settled_count(self):
        """Settled count must remain frozen and stop producing settle logs."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        tracker = lock_tracker_for_settling(bldc)
        sample_time = tracker.actual_stop_print_time
        for count in (20.0, 20.0, 20.0):
            sample_time += 0.04
            bldc.reactor.pause(sample_time)
            bldc.tachometer.handle_tachometer(sample_time, count, sample_time)
        sample_time += 0.04
        bldc.reactor.pause(sample_time)
        bldc.tachometer.handle_tachometer(sample_time, 30.0, sample_time)
        settle_log_count = sum(
            ("BLDC_POSITION_SETTLE" in message for message in bldc.mmu.stepper_logs)
        )
        self.assertApproxEqual(
            (
                tracker.learning_completed,
                tracker.settled_count,
                bldc.get_last_move_revolutions(75.0),
                settle_log_count,
            ),
            (True, 20.0, 20.0 / 18.0, 3),
        )

    def test_zero_rpm_callbacks_with_stale_edge_time_finalize_learning(self):
        """No new edge exists at standstill, so callback time must prove freshness."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        tracker = lock_tracker_for_settling(bldc)
        edge_time = tracker.actual_stop_print_time + 0.01
        bldc.tachometer.last_tach_rpm = 0.0
        for sample_offset in (0.04, 0.08):
            sample_time = tracker.actual_stop_print_time + sample_offset
            bldc.handle_tachometer_position_sample(sample_time, 20.0, edge_time)
        self.assertApproxEqual(
            (tracker.standstill_sample_count, tracker.learning_completed, tracker.settled_count),
            (2, True, 20.0),
        )

    def test_tach_extension_does_not_release_nominal_stop_reservation(self):
        """A transient late endpoint must not let PWM block a later earlier stop."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc.mcu._min_schedule = 0.1
        bldc.min_schedule_time = 0.11
        bldc.direction_setup_time = 0.1
        bldc.motion_sample_time = 0.04
        bldc.last_dir = 1
        bldc.set_rotation_distance(12.6846)
        prime_tachometer_count(bldc)
        bldc.start_move(100.0, 600.0)
        tracker = bldc._finite_move_tracker
        nominal_stop_time = tracker.stop_descriptor.print_time
        bldc._stop_motion_timer()
        bldc._schedule_finite_move_endpoint(nominal_stop_time + 0.2)
        self.assertIs(bldc._is_endpoint_reserved(nominal_stop_time - 0.01), True)

    def test_first_move_uses_recent_frequency_for_live_stop_lead(self):
        """Unlearned moves must command PWM off before target to absorb coast."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc.set_rotation_distance(7.5)
        prime_tachometer_count(bldc)
        bldc.start_move(75.0, 600.0)
        tracker = bldc._finite_move_tracker
        bldc._stop_motion_timer()
        sample_time = tracker.start_print_time + 0.5
        bldc.tachometer.handle_tachometer(sample_time, 50.0, sample_time)
        self.assertApproxEqual(
            tracker.stop_lead_counts, 100.0 * bldc.tachometer.tachometer_sample_time
        )

    def test_frequency_seeded_move_keeps_learned_stop_lead(self):
        """Settled frequency already includes coast, avoiding duplicate live lead."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc.set_rotation_distance(7.5)
        feed_forward = bldc._position_feed_forward[1]
        feed_forward.learn_frequency(100.0, 4800.0)
        feed_forward.stop_lead_counts = 3.0
        prime_tachometer_count(bldc)
        bldc.start_move(75.0, 600.0)
        tracker = bldc._finite_move_tracker
        bldc._stop_motion_timer()
        sample_time = tracker.start_print_time + 0.5
        bldc.tachometer.handle_tachometer(sample_time, 50.0, sample_time)
        self.assertApproxEqual(tracker.stop_lead_counts, 3.0)

    def test_reversal_plans_direction_setup_before_position_start(self):
        """Plan explicit DIR setup time before a reversed move begins."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc.last_dir = 1
        prime_tachometer_count(bldc)
        bldc.start_move(-100.0, 600.0)
        bldc.reactor.pause(bldc.motion_sample_time)
        tracker = bldc._finite_move_tracker
        direction_time = bldc.mcu_dir_pin.digital_calls[-1][0]
        pwm_time = bldc.mcu_pwm_pin.pwm_calls[0][0]
        self.assertApproxEqual(
            (tracker.start_print_time - direction_time, pwm_time),
            (bldc.direction_setup_time, tracker.start_print_time),
        )

    def test_mapped_finite_move_keeps_direction_through_kick_1(self):
        """Gate mapping must remain stable from finite-move kick through cruise."""
        mapped_gate_bldc = _make_mapped_gate_bldc(self.bldc_module)
        gate = 0
        distance = 100.0
        expected_direction = 1
        bldc = mapped_gate_bldc
        bldc.mmu.gate_selected = gate
        prime_tachometer_count(bldc)
        bldc.start_move(distance, 20.0)
        cruise_dispatch_time = bldc._finite_move_tracker.cruise_start_time + bldc.motion_sample_time
        advance_motion_through(bldc, cruise_dispatch_time)
        self.assertEqual(direction_pin_values(bldc), [expected_direction])

    def test_mapped_finite_move_keeps_direction_through_kick_2(self):
        """Gate mapping must remain stable from finite-move kick through cruise."""
        mapped_gate_bldc = _make_mapped_gate_bldc(self.bldc_module)
        gate = 0
        distance = -100.0
        expected_direction = 0
        bldc = mapped_gate_bldc
        bldc.mmu.gate_selected = gate
        prime_tachometer_count(bldc)
        bldc.start_move(distance, 20.0)
        cruise_dispatch_time = bldc._finite_move_tracker.cruise_start_time + bldc.motion_sample_time
        advance_motion_through(bldc, cruise_dispatch_time)
        self.assertEqual(direction_pin_values(bldc), [expected_direction])

    def test_mapped_finite_move_keeps_direction_through_kick_3(self):
        """Gate mapping must remain stable from finite-move kick through cruise."""
        mapped_gate_bldc = _make_mapped_gate_bldc(self.bldc_module)
        gate = 1
        distance = 100.0
        expected_direction = 0
        bldc = mapped_gate_bldc
        bldc.mmu.gate_selected = gate
        prime_tachometer_count(bldc)
        bldc.start_move(distance, 20.0)
        cruise_dispatch_time = bldc._finite_move_tracker.cruise_start_time + bldc.motion_sample_time
        advance_motion_through(bldc, cruise_dispatch_time)
        self.assertEqual(direction_pin_values(bldc), [expected_direction])

    def test_mapped_finite_move_keeps_direction_through_kick_4(self):
        """Gate mapping must remain stable from finite-move kick through cruise."""
        mapped_gate_bldc = _make_mapped_gate_bldc(self.bldc_module)
        gate = 1
        distance = -100.0
        expected_direction = 1
        bldc = mapped_gate_bldc
        bldc.mmu.gate_selected = gate
        prime_tachometer_count(bldc)
        bldc.start_move(distance, 20.0)
        cruise_dispatch_time = bldc._finite_move_tracker.cruise_start_time + bldc.motion_sample_time
        advance_motion_through(bldc, cruise_dispatch_time)
        self.assertEqual(direction_pin_values(bldc), [expected_direction])

    def test_mapped_open_move_keeps_direction_through_kick_1(self):
        """Gate mapping must remain stable from open-move kick through cruise."""
        mapped_gate_bldc = _make_mapped_gate_bldc(self.bldc_module)
        gate = 0
        speed = 20.0
        expected_direction = 1
        bldc = mapped_gate_bldc
        bldc.mmu.gate_selected = gate
        bldc.start_move(None, speed)
        cruise_dispatch_time = (
            bldc.min_schedule_time
            + bldc.direction_setup_time
            + bldc.kick_start_time
            + bldc.motion_sample_time
        )
        advance_motion_through(bldc, cruise_dispatch_time)
        self.assertEqual(direction_pin_values(bldc), [expected_direction])

    def test_mapped_open_move_keeps_direction_through_kick_2(self):
        """Gate mapping must remain stable from open-move kick through cruise."""
        mapped_gate_bldc = _make_mapped_gate_bldc(self.bldc_module)
        gate = 0
        speed = -20.0
        expected_direction = 0
        bldc = mapped_gate_bldc
        bldc.mmu.gate_selected = gate
        bldc.start_move(None, speed)
        cruise_dispatch_time = (
            bldc.min_schedule_time
            + bldc.direction_setup_time
            + bldc.kick_start_time
            + bldc.motion_sample_time
        )
        advance_motion_through(bldc, cruise_dispatch_time)
        self.assertEqual(direction_pin_values(bldc), [expected_direction])

    def test_mapped_open_move_keeps_direction_through_kick_3(self):
        """Gate mapping must remain stable from open-move kick through cruise."""
        mapped_gate_bldc = _make_mapped_gate_bldc(self.bldc_module)
        gate = 1
        speed = 20.0
        expected_direction = 0
        bldc = mapped_gate_bldc
        bldc.mmu.gate_selected = gate
        bldc.start_move(None, speed)
        cruise_dispatch_time = (
            bldc.min_schedule_time
            + bldc.direction_setup_time
            + bldc.kick_start_time
            + bldc.motion_sample_time
        )
        advance_motion_through(bldc, cruise_dispatch_time)
        self.assertEqual(direction_pin_values(bldc), [expected_direction])

    def test_mapped_open_move_keeps_direction_through_kick_4(self):
        """Gate mapping must remain stable from open-move kick through cruise."""
        mapped_gate_bldc = _make_mapped_gate_bldc(self.bldc_module)
        gate = 1
        speed = -20.0
        expected_direction = 1
        bldc = mapped_gate_bldc
        bldc.mmu.gate_selected = gate
        bldc.start_move(None, speed)
        cruise_dispatch_time = (
            bldc.min_schedule_time
            + bldc.direction_setup_time
            + bldc.kick_start_time
            + bldc.motion_sample_time
        )
        advance_motion_through(bldc, cruise_dispatch_time)
        self.assertEqual(direction_pin_values(bldc), [expected_direction])

    def test_unchanged_direction_and_pwm_share_queue_slot(self):
        """Avoid consuming a separate MCU queue interval for unchanged DIR."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc.last_dir = 0
        prime_tachometer_count(bldc)
        bldc.start_move(-100.0, 600.0)
        tracker = bldc._finite_move_tracker
        self.assertApproxEqual(bldc.mcu_pwm_pin.pwm_calls[0][0], tracker.start_print_time)

    def test_bldc_units_on_same_mcu_reuse_pin_request_queue(self):
        """Serialize all BLDC units on one MCU through one request queue."""
        bldc_module = self.bldc_module
        printer = FakePrinter(toolhead=FakeToolhead(extruder=FakeExtruder()))
        mmu = FakeMmu(toolhead=printer.lookup_object("toolhead"), gate_selected=0)

        def create_unit(unit):
            return bldc_module.MmuGearBldc(
                FakeConfig(
                    printer,
                    {
                        "dir_pin": "D%d" % unit,
                        "pwm_pin": "P%d" % unit,
                        "direction_map": [0],
                        "tachometer_pin": "T%d" % unit,
                    },
                    name="mmu_gear_bldc unit%d" % unit,
                ),
                mmu,
            )

        first = create_unit(0)
        second = create_unit(1)
        self.assertIs(first.pin_request_queue, second.pin_request_queue)

    def test_finite_move_wait_includes_scheduled_start_lead(self):
        """Callers must not finish waiting before queued BLDC motion reaches its stop."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        move_wait = bldc_runtime.start_move(10.0, 20.0)
        self.assertApproxEqual(
            move_wait, bldc_runtime.min_schedule_time + bldc_runtime.direction_setup_time + 0.5
        )

    def test_finite_move_stop_deadline_uses_effective_max_speed(self):
        """A maximum-RPM clamp must extend runtime so requested distance remains possible."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc.set_rotation_distance(0.0016)
        effective_speed = bldc.get_effective_max_linear_speed()
        move_wait = bldc.start_move(100.0, 600.0)
        self.assertApproxEqual(
            move_wait, bldc.min_schedule_time + bldc.direction_setup_time + 100.0 / effective_speed
        )

    def test_rpm_limited_finite_move_stops_at_effective_deadline(self):
        """Queued stop must follow effective motor speed, not impossible requested speed."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc = bldc_runtime
        bldc.set_rotation_distance(0.0016)
        move_wait = bldc.start_move(100.0, 600.0)
        stop = next(
            (
                descriptor
                for descriptor, _source in bldc.motion_queue
                if type(descriptor) is bldc_module.MotionStop
            )
        )
        self.assertApproxEqual(stop.print_time, bldc._get_scheduled_print_time() + move_wait)

    def test_finite_move_limits_speed_and_extends_stop_deadline_1(self):
        """Both directions must preserve distance when maximum RPM limits speed."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        distance = 100.0
        bldc = bldc_runtime
        max_speed = bldc.get_effective_max_rpm() * bldc.rotation_distance / 60.0
        requested_speed = max_speed * 2.0
        move_wait = bldc.start_move(distance, requested_speed)
        cruise = next(
            (
                descriptor
                for descriptor, _source in bldc.motion_queue
                if type(descriptor) is bldc_module.MotionDescriptor
            )
        )
        self.assertApproxEqual(abs(cruise.speed_mm_s), max_speed)
        self.assertApproxEqual(
            move_wait,
            bldc.min_schedule_time + bldc.direction_setup_time + abs(distance) / max_speed,
        )
        self.assertEqual(len(bldc.mmu.warning_logs), 1)

    def test_finite_move_limits_speed_and_extends_stop_deadline_2(self):
        """Both directions must preserve distance when maximum RPM limits speed."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        distance = -100.0
        bldc = bldc_runtime
        max_speed = bldc.get_effective_max_rpm() * bldc.rotation_distance / 60.0
        requested_speed = max_speed * 2.0
        move_wait = bldc.start_move(distance, requested_speed)
        cruise = next(
            (
                descriptor
                for descriptor, _source in bldc.motion_queue
                if type(descriptor) is bldc_module.MotionDescriptor
            )
        )
        self.assertApproxEqual(abs(cruise.speed_mm_s), max_speed)
        self.assertApproxEqual(
            move_wait,
            bldc.min_schedule_time + bldc.direction_setup_time + abs(distance) / max_speed,
        )
        self.assertEqual(len(bldc.mmu.warning_logs), 1)

    def test_finite_move_uses_calibrated_minimum_speed_and_matching_deadline(self):
        """Below-map requests must not overtravel at the mapped minimum motor RPM."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc = bldc_runtime
        bldc.set_calibration_map(
            {
                "pwm_min": 0.3,
                "points": [
                    {"pwm": 0.3, "rpm": 1200.0},
                    {"pwm": 0.6, "rpm": 2400.0},
                    {"pwm": 1.0, "rpm": 3600.0},
                ],
            }
        )
        min_speed = 1200.0 * bldc.rotation_distance / 60.0
        move_wait = bldc.start_move(100.0, min_speed / 4.0)
        cruise = next(
            (
                descriptor
                for descriptor, _source in bldc.motion_queue
                if type(descriptor) is bldc_module.MotionDescriptor
            )
        )
        self.assertApproxEqual(
            (abs(cruise.speed_mm_s), move_wait),
            (min_speed, bldc.min_schedule_time + bldc.direction_setup_time + 100.0 / min_speed),
        )

    def test_synced_move_below_minimum_uses_fixed_torque_assist(self):
        """Low-speed extruder sync must apply capped torque without speed PID."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc = bldc_runtime
        set_test_calibration_map(bldc)
        move = FakeMove(
            [0.0, 0.0, 0.0, 1.0],
            start_v=0.0,
            cruise_v=5.0,
            end_v=0.0,
            accel=100.0,
            accel_t=0.05,
            cruise_t=0.5,
            decel_t=0.05,
        )
        bldc.queue_trapzoid_move(move, 1.0, 10.0, "process_move_push")
        descriptor, source = bldc.motion_queue[0]
        actual = (
            len(bldc.motion_queue),
            isinstance(descriptor, bldc_module.MotionTorqueAssist),
            descriptor.pwm,
            descriptor.duration,
            descriptor.pid_enable,
            source,
        )
        self.assertEqual(actual[0], 1)
        self.assertEqual(actual[1], True)
        self.assertApproxEqual(actual[2], 0.2)
        self.assertApproxEqual(actual[3], 0.6)
        self.assertEqual(actual[4], False)
        self.assertEqual(actual[5], "process_move_push")

    def test_sync_monitor_ignores_retraction_below_configured_distance(self):
        """Short slicer retractions must not reverse the BLDC assist motor."""
        bldc_module = self.bldc_module
        bldc, monitor = _create_sync_monitor(bldc_module, 1.0)
        move = _create_extruder_move(-0.8)
        monitor._handle_process_move(10.0, move, 3)
        self.assertEqual(bldc.motion_queue, [])

    def test_sync_monitor_follows_retraction_at_configured_distance(self):
        """Retractions meeting the threshold must retain reverse-assist behavior."""
        bldc_module = self.bldc_module
        bldc, monitor = _create_sync_monitor(bldc_module, 1.0)
        move = _create_extruder_move(-1.0)
        monitor._handle_process_move(10.0, move, 3)
        self.assertEqual(len(bldc.motion_queue), 3)

    def test_sync_monitor_zero_distance_disables_retraction_damping(self):
        """Zero threshold must preserve follow-every-retraction compatibility."""
        bldc_module = self.bldc_module
        bldc, monitor = _create_sync_monitor(bldc_module, 0.0)
        move = _create_extruder_move(-0.1)
        monitor._handle_process_move(10.0, move, 3)
        self.assertEqual(len(bldc.motion_queue), 3)

    def test_sync_monitor_never_damps_forward_extrusion(self):
        """Retraction damping must not suppress ordinary extrusion assistance."""
        bldc_module = self.bldc_module
        bldc, monitor = _create_sync_monitor(bldc_module, 1.0)
        move = _create_extruder_move(0.1)
        monitor._handle_process_move(10.0, move, 3)
        self.assertEqual(len(bldc.motion_queue), 3)

    def test_synced_move_at_minimum_keeps_speed_control(self):
        """Minimum achievable speed remains ordinary mapped tachometer control."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc = bldc_runtime
        set_test_calibration_map(bldc)
        minimum_speed = bldc.get_effective_min_linear_speed()
        move = FakeMove(
            [0.0, 0.0, 0.0, 1.0],
            start_v=0.0,
            cruise_v=minimum_speed,
            end_v=0.0,
            accel=100.0,
            accel_t=0.05,
            cruise_t=0.5,
            decel_t=0.05,
        )
        bldc.queue_trapzoid_move(move, 1.0, 10.0, "process_move_push")
        self.assertTrue(
            not any(
                (
                    isinstance(descriptor, bldc_module.MotionTorqueAssist)
                    for descriptor, _source in bldc.motion_queue
                )
            )
        )

    def test_open_ended_low_speed_move_uses_torque_assist_without_kick(self):
        """Gear-plus-extruder assist must never apply the destructive full-PWM kick."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc = bldc_runtime
        set_test_calibration_map(bldc)
        move_wait = bldc.start_move(None, 5.0)
        descriptor, source = bldc.motion_queue[0]
        actual = (
            move_wait,
            len(bldc.motion_queue),
            isinstance(descriptor, bldc_module.MotionTorqueAssist),
            descriptor.pwm,
            descriptor.duration,
            source,
        )
        self.assertEqual(actual[0], None)
        self.assertEqual(actual[1], 1)
        self.assertEqual(actual[2], True)
        self.assertApproxEqual(actual[3], 0.2)
        self.assertEqual(actual[4], bldc_module.INFINITY)
        self.assertEqual(actual[5], "move")

    def test_open_ended_move_has_no_finite_wait(self):
        """Event-driven callers own the stop signal and therefore have no fixed wait."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        move_wait = bldc_runtime.start_move(None, 20.0)
        self.assertIs(move_wait, None)

    def test_calibration_uses_first_moving_sample_as_mapped_pwm_floor(self):
        """A stalled low-PWM sample must not become the mapped minimum motor drive."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime

        def apply_calibration_pwm(pwm, _print_time=None):
            bldc.tachometer.last_tach_rpm = 0.0 if pwm < 0.3 else pwm * 1000.0

        bldc._safe_set_pwm = apply_calibration_pwm
        bldc.tachometer.is_tach_sample_valid = lambda: (bldc.tachometer.last_tach_rpm > 0.0, "ok")
        payload = bldc.calibrate_pwm_rpm_map(9)
        self.assertApproxEqual(
            (len(payload["points"]), payload["points"][0]["pwm"], bldc.rpm_to_pwm(1.0)),
            (9, 0.33, 0.33),
        )

    def test_calibration_startup_margin_rounds_up(self):
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        self.assertApproxEqual(bldc_runtime._apply_startup_pwm_margin(0.3992), 0.43)

    def test_calibration_startup_margin_clamps_to_pwm_max(self):
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        self.assertApproxEqual(bldc_runtime._apply_startup_pwm_margin(0.99), 1.0)

    def test_first_map_point_uses_extended_sampling_window(self):
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc._safe_set_pwm = lambda _pwm, _print_time=None: None

        def update_rpm():
            elapsed = bldc.reactor.monotonic() - bldc.CALIBRATION_DEFAULT_SETTLE_S
            bldc.tachometer.last_tach_rpm = 0.0 if elapsed < 1.0 else 120.0
            return (bldc.tachometer.last_tach_rpm > 0.0, "ok")

        bldc.tachometer.is_tach_sample_valid = update_rpm
        point = bldc._sample_first_map_point(0.3992)
        self.assertApproxEqual(point["rpm"], 120.0)

    def test_calibration_fails_without_positive_rpm_sample(self):
        """Calibration must fail clearly when no PWM value starts the motor."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc_runtime._safe_set_pwm = lambda _pwm, _print_time=None: None
        bldc_runtime.tachometer.is_tach_sample_valid = lambda: (False, "tach_zero")
        with self.assertRaisesRegex(bldc_module.MmuError, "no positive RPM detected"):
            bldc_runtime.calibrate_pwm_rpm_map(3)

    def test_map_sampling_extends_rpm_drop_until_reading_recovers(self):
        """A transient RPM drop gets the longer sampling window before acceptance."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime

        def update_rpm():
            sample_elapsed = bldc.reactor.monotonic() - bldc.CALIBRATION_DEFAULT_SETTLE_S
            bldc.tachometer.last_tach_rpm = 80.0 if sample_elapsed < 1.0 else 120.0
            return (True, "ok")

        bldc._safe_set_pwm = lambda _pwm, _print_time=None: None
        bldc.tachometer.is_tach_sample_valid = update_rpm
        point = bldc._sample_calibration_window(0.4, previous_positive_rpm=100.0)
        self.assertApproxEqual(point["rpm"], 120.0)
        self.assertApproxEqual(bldc.reactor.monotonic(), 0.35 + 3.0)

    def test_map_sampling_equal_rpm_uses_normal_window(self):
        """An equal RPM point must not spend extra time in adaptive sampling."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc._safe_set_pwm = lambda _pwm, _print_time=None: None
        bldc.tachometer.last_tach_rpm = 100.0
        bldc.tachometer.is_tach_sample_valid = lambda: (True, "ok")
        point = bldc._sample_calibration_window(0.4, previous_positive_rpm=100.0)
        self.assertApproxEqual(point["rpm"], 100.0)
        self.assertApproxEqual(bldc.reactor.monotonic(), 0.35 + 1.0)

    def test_map_sampling_persistent_zero_returns_zero_after_extension(self):
        """A zero after positive motion receives extension time but remains invalid."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc._safe_set_pwm = lambda _pwm, _print_time=None: None
        bldc.tachometer.last_tach_rpm = 0.0
        bldc.tachometer.is_tach_sample_valid = lambda: (False, "tach_zero")
        point = bldc._sample_calibration_window(0.4, previous_positive_rpm=100.0)
        self.assertApproxEqual(point["rpm"], 0.0)
        self.assertApproxEqual(bldc.reactor.monotonic(), 0.35 + 3.0)

    def test_calibration_map_uses_strict_payload_and_interpolates(self):
        """Current maps carry the runtime floor and provide smooth lookup."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        ok, reason = bldc_runtime.set_calibration_map(
            {
                "pwm_min": 0.3,
                "points": [
                    {"pwm": 0.3, "rpm": 100.0},
                    {"pwm": 0.4, "rpm": 150.0},
                    {"pwm": 0.6, "rpm": 200.0},
                ],
            }
        )
        self.assertEqual((ok, reason), (True, None))
        self.assertEqual(
            bldc_runtime.calibration_map_points,
            [{"pwm": 0.3, "rpm": 100.0}, {"pwm": 0.4, "rpm": 150.0}, {"pwm": 0.6, "rpm": 200.0}],
        )
        self.assertApproxEqual(bldc_runtime.get_effective_max_rpm(), 200.0)
        self.assertApproxEqual(bldc_runtime.rpm_to_pwm(125.0), 0.35)
        self.assertApproxEqual(bldc_runtime.rpm_to_pwm(50.0), 0.3)
        self.assertApproxEqual(bldc_runtime.rpm_to_pwm(250.0), 0.6)

    def test_multi_pass_calibration_discovers_all_floors_before_aggregating(self):
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        discovery_floors = iter([0.3, 0.4, 0.35, 0.38])
        pass_rpms = iter(
            [
                [100.0, 200.0, 300.0, 400.0],
                [100.0, 220.0, 320.0, 400.0],
                [100.0, 210.0, 310.0, 400.0],
                [100.0, 230.0, 330.0, 400.0],
            ]
        )
        events = []
        progress = []
        bldc._discover_startup_pwm = lambda _point_count: events.append("discover") or next(
            discovery_floors
        )
        bldc._wait_for_calibration_standstill = lambda: events.append("standstill")

        def sample_map_point(pwm, _previous_rpm=None):
            if not events or events[-1] == "standstill":
                sample_map_point.rpms = next(pass_rpms)
            point_index = len([event for event in events if event == "sample"])
            events.append("sample")
            return {"pwm": pwm, "rpm": sample_map_point.rpms[point_index % 4]}

        bldc._sample_calibration_window = sample_map_point
        bldc._safe_set_direction = lambda _forward: None
        bldc.stop = lambda: None
        bldc.mmu.movequeue_wait = lambda: None
        payload = bldc.calibrate_pwm_rpm_map(4, passes=4, progress_callback=progress.append)
        self.assertEqual(
            payload,
            {
                "pwm_min": 0.43,
                "points": [
                    {"pwm": 0.43, "rpm": 100.0},
                    {"pwm": 0.62, "rpm": 215.0},
                    {"pwm": 0.81, "rpm": 315.0},
                    {"pwm": 1.0, "rpm": 400.0},
                ],
            },
        )
        self.assertEqual(events[:3], ["discover", "discover", "discover"])
        self.assertGreater(events.index("standstill"), events.index("discover"))
        self.assertEqual(
            progress[:4],
            [
                "BLDC discovery pass 1/4: pwm_min=0.300",
                "BLDC discovery pass 2/4: pwm_min=0.400",
                "BLDC discovery pass 3/4: pwm_min=0.350",
                "BLDC discovery pass 4/4: pwm_min=0.380",
            ],
        )
        self.assertEqual(progress[-1], "BLDC calibration complete: pwm_min=0.430, discarded=0")

    def test_calibration_reports_rpm_drop_after_completing_sweep(self):
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc = bldc_runtime
        sampled_rpms = iter([100.0, 200.0, 150.0, 300.0])
        progress = []
        bldc._discover_startup_pwm = lambda _point_count: 0.3
        bldc._wait_for_calibration_standstill = lambda: None
        bldc._sample_first_map_point = lambda pwm: {"pwm": pwm, "rpm": next(sampled_rpms)}
        bldc._sample_calibration_window = lambda pwm, _previous_rpm: {
            "pwm": pwm,
            "rpm": next(sampled_rpms),
        }
        bldc._safe_set_direction = lambda _forward: None
        bldc.stop = lambda: None
        bldc.mmu.movequeue_wait = lambda: None
        with self.assertRaisesRegex(
            bldc_module.MmuError, "Check tachometer_ppr configuration; value may be too high"
        ):
            bldc.calibrate_pwm_rpm_map(4, passes=1, progress_callback=progress.append)
        self.assertIn("|    3 |    0.777 |    150.0 |    ERROR |", progress[-1])
        self.assertIn("|    4 |    1.000 |    300.0 |       OK |", progress[-1])

    def test_calibration_allows_equal_rpm_points(self):
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        self.assertTrue(not bldc_runtime._is_calibration_rpm_drop(200.0, 200.0))

    def test_multi_pass_calibration_rejects_out_of_range_pass_count_1(self):
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        passes = 0
        with self.assertRaisesRegex(bldc_module.MmuError, "passes must be between 1 and 10"):
            bldc_runtime.calibrate_pwm_rpm_map(3, passes=passes)

    def test_multi_pass_calibration_rejects_out_of_range_pass_count_2(self):
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        passes = 11
        with self.assertRaisesRegex(bldc_module.MmuError, "passes must be between 1 and 10"):
            bldc_runtime.calibrate_pwm_rpm_map(3, passes=passes)

    def test_motion_timer_keeps_active_descriptor_until_duration_expires(self):
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc = bldc_runtime
        bldc.motion_state = bldc.MOTION_STATE_MOVING
        descriptor = bldc_module.MotionDescriptor(
            active_print_time(bldc) - 0.01, 10.0, True, 1, duration=1.0
        )
        bldc.motion_queue = [(descriptor, "move")]
        capture_pwm_requests(bldc)
        bldc._motion_timer_callback(bldc.reactor.monotonic())
        self.assertEqual(bldc.motion_queue, [(descriptor, "move")])
        bldc.reactor.pause(2.0)
        bldc._motion_timer_callback(bldc.reactor.monotonic())
        self.assertEqual(bldc.motion_queue, [])

    def test_motion_timer_wakes_at_next_descriptor_commit_boundary(self):
        """Wake exactly when the future stop enters the safe commit horizon."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc = bldc_runtime
        bldc.min_schedule_time = 0.11
        bldc.motion_sample_time = 0.04
        bldc.motion_state = bldc.MOTION_STATE_MOVING
        bldc.motion_queue = [
            (bldc_module.MotionDescriptor(0.0, 10.0, True, 1, duration=1.0), "move"),
            (bldc_module.MotionStop(0.13), "move"),
        ]
        wake = bldc._motion_timer_callback(0.0)
        self.assertApproxEqual(wake, 0.02)

    def test_motion_timer_requeues_pwm_when_tach_correction_changes(self):
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc = bldc_runtime
        bldc.motion_state = bldc.MOTION_STATE_MOVING
        bldc.motion_queue = [
            (
                bldc_module.MotionDescriptor(
                    active_print_time(bldc) - 0.01, 10.0, True, 1, duration=1.0
                ),
                "move",
            )
        ]
        prepare_fresh_tachometer(bldc)
        bldc.tachometer.control_correction_pwm = 0.05
        pwm_requests = capture_pwm_requests(bldc)
        bldc._motion_timer_callback(bldc.reactor.monotonic())
        bldc.tachometer.control_correction_pwm = 0.09
        bldc.reactor.pause(0.02)
        bldc._motion_timer_callback(bldc.reactor.monotonic())
        self.assertEqual([round(value, 4) for _, value in pwm_requests], [0.2936, 0.3336])

    def test_motion_timer_dispatches_torque_assist_without_pid_correction(self):
        """Stall assist must remain at its hard PWM cap even with stale PID state."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc = bldc_runtime
        bldc.motion_state = bldc.MOTION_STATE_MOVING
        bldc.motion_queue = [
            (
                bldc_module.MotionTorqueAssist(
                    active_print_time(bldc) - 0.01, bldc.torque_assist_pwm, 1.0, 1
                ),
                "process_move_push",
            )
        ]
        bldc.tachometer.enabled = True
        bldc.tachometer.control_correction_pwm = 0.2
        bldc.tachometer.integral_correction_pwm = 0.1
        pwm_requests = capture_pwm_requests(bldc)
        bldc._motion_timer_callback(bldc.reactor.monotonic())
        self.assertApproxEqual(
            (
                pwm_requests[-1][1],
                bldc.tachometer.enabled,
                bldc.tachometer.control_correction_pwm,
                bldc.tachometer.integral_correction_pwm,
            ),
            (0.2, False, 0.0, 0.0),
        )
        self.assertEqual(bldc.commanded_source, "torque_assist")

    def test_torque_assist_stops_during_gap_before_future_extrusion(self):
        """Torque must drop while extruder is stationary between queued moves."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc = bldc_runtime
        bldc.sync_active = True
        bldc.motion_state = bldc.MOTION_STATE_MOVING
        bldc.last_pwm = bldc.last_effective_pwm = bldc.torque_assist_pwm
        future_assist = bldc_module.MotionTorqueAssist(
            active_print_time(bldc) + 1.0, bldc.torque_assist_pwm, 1.0, 1
        )
        bldc.motion_queue = [(future_assist, "process_move_push")]
        pwm_requests = capture_pwm_requests(bldc)
        bldc._motion_timer_callback(bldc.reactor.monotonic())
        self.assertApproxEqual(pwm_requests[-1][1], 0.0)
        self.assertEqual(bldc.motion_queue, [(future_assist, "process_move_push")])

    def test_first_pid_enabled_dispatch_applies_existing_tach_correction(self):
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc = bldc_runtime
        bldc.motion_state = bldc.MOTION_STATE_MOVING
        bldc.motion_queue = [
            (
                bldc_module.MotionDescriptor(
                    active_print_time(bldc) - 0.01, 10.0, True, 1, duration=1.0
                ),
                "move",
            )
        ]
        prepare_fresh_tachometer(bldc)
        bldc.tachometer.enabled = False
        bldc.tachometer.control_correction_pwm = 0.05
        pwm_requests = capture_pwm_requests(bldc)
        bldc._motion_timer_callback(bldc.reactor.monotonic())
        self.assertApproxEqual(pwm_requests[-1][1], 0.293636, abs_tol=1e-06)

    def test_motion_timer_skips_small_pwm_correction_delta(self):
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc = bldc_runtime
        bldc.motion_state = bldc.MOTION_STATE_MOVING
        bldc.motion_queue = [
            (
                bldc_module.MotionDescriptor(
                    active_print_time(bldc) - 0.01, 10.0, True, 1, duration=1.0
                ),
                "move",
            )
        ]
        prepare_fresh_tachometer(bldc)
        bldc.tachometer.control_correction_pwm = 0.05
        pwm_requests = capture_pwm_requests(bldc)
        bldc._motion_timer_callback(bldc.reactor.monotonic())
        bldc.tachometer.control_correction_pwm = 0.051
        bldc.reactor.pause(0.02)
        bldc._motion_timer_callback(bldc.reactor.monotonic())
        self.assertEqual(len(pwm_requests), 1)

    def test_motion_timer_stop_descriptor_sends_zero_and_stops_timer(self):
        """A finite stop must clear velocity-control state before another move starts."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc = bldc_runtime
        bldc.motion_state = bldc.MOTION_STATE_MOVING
        bldc.last_pwm = 0.5
        bldc.last_effective_pwm = 0.5
        bldc.commanded_rpm = 600.0
        bldc.commanded_source = "move"
        bldc.tachometer.enabled = True
        bldc.tachometer.set_commanded(600.0, "move")
        bldc.tachometer.integral_correction_pwm = 0.1
        bldc.motion_queue = [(bldc_module.MotionStop(active_print_time(bldc) - 0.01), "move")]
        pwm_requests = capture_pwm_requests(bldc)
        wake = bldc._motion_timer_callback(bldc.reactor.monotonic())
        self.assertApproxEqual(
            (
                wake,
                pwm_requests[-1][1],
                bldc.motion_state,
                bldc.commanded_rpm,
                bldc.tachometer.enabled,
                bldc.tachometer.commanded_rpm,
                bldc.tachometer.integral_correction_pwm,
            ),
            (bldc.reactor.NEVER, 0.0, bldc.MOTION_STATE_STOP, 0.0, False, 0.0, 0.0),
        )

    def test_start_move_resets_stale_pid_correction(self):
        """Every standalone move must start without correction inherited from an earlier move."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc.commanded_rpm = 600.0
        bldc.tachometer.enabled = True
        bldc.tachometer.set_commanded(600.0, "move")
        bldc.tachometer.control_correction_pwm = 0.15
        bldc.tachometer.integral_correction_pwm = 0.1
        bldc.start_move(100.0, 20.0)
        self.assertApproxEqual(
            (
                bldc.commanded_rpm,
                bldc.tachometer.enabled,
                bldc.tachometer.commanded_rpm,
                bldc.tachometer.control_correction_pwm,
                bldc.tachometer.integral_correction_pwm,
            ),
            (0.0, False, 0.0, 0.0, 0.0),
        )

    def test_finite_move_uses_tach_counts_to_advance_endpoint(self):
        """Cumulative tach travel must move endpoint earlier when motor runs faster than planned."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc = bldc_runtime
        prime_tachometer_count(bldc)
        bldc.start_move(100.0, 20.0)
        original_stop_time = next(
            (
                descriptor.print_time
                for descriptor, _source in bldc.motion_queue
                if type(descriptor) is bldc_module.MotionStop
            )
        )
        sample_time = active_print_time(bldc) + 2.0
        bldc.tachometer.handle_tachometer(sample_time, 450.0, sample_time)
        updated_stop_time = next(
            (
                descriptor.print_time
                for descriptor, _source in bldc.motion_queue
                if type(descriptor) is bldc_module.MotionStop
            )
        )
        self.assertLess(updated_stop_time, original_stop_time)

    def test_tach_endpoint_update_wakes_motion_timer(self):
        """Reschedule the motion timer immediately after endpoint correction."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        prime_tachometer_count(bldc)
        bldc.start_move(100.0, 20.0)
        bldc.motion_timer["when"] = 10.0
        sample_time = bldc._finite_move_tracker.start_print_time + 2.0
        bldc.tachometer.handle_tachometer(sample_time, 450.0, sample_time)
        self.assertLess(bldc.motion_timer["when"], 10.0)

    def test_delayed_tach_sample_feeds_unreported_motion_into_stop_time(self):
        """MCU edge time must prevent callback latency from adding extra motor travel."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc.set_rotation_distance(7.5)
        prime_tachometer_count(bldc)
        bldc.start_move(75.0, 600.0)
        tracker = bldc._finite_move_tracker
        count_time = tracker.start_print_time + 0.5
        callback_time = count_time + 0.3
        bldc.tachometer.handle_tachometer(callback_time, 70.0, count_time)
        frequency = 140.0
        stop_lead_counts = frequency * bldc.tachometer.tachometer_sample_time
        predicted_callback_count = 70.0 + frequency * (callback_time - count_time)
        expected_stop_time = (
            callback_time
            + (tracker.target_count_delta - stop_lead_counts - predicted_callback_count) / frequency
        )
        self.assertApproxEqual(tracker.stop_descriptor.print_time, expected_stop_time)

    def test_recent_tach_rate_moves_endpoint_later_after_slowdown(self):
        """Recent loaded RPM must outweigh an obsolete fast whole-move average."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc.set_rotation_distance(7.5)
        prime_tachometer_count(bldc)
        bldc.start_move(75.0, 600.0)
        tracker = bldc._finite_move_tracker
        bldc.tachometer.handle_tachometer(
            tracker.start_print_time + 0.5, 50.0, tracker.start_print_time + 0.5
        )
        bldc.tachometer.handle_tachometer(
            tracker.start_print_time + 1.0, 100.0, tracker.start_print_time + 1.0
        )
        bldc.tachometer.handle_tachometer(
            tracker.start_print_time + 1.5, 125.0, tracker.start_print_time + 1.5
        )
        filtered_frequency = 75.0
        stop_lead_counts = filtered_frequency * bldc.tachometer.tachometer_sample_time
        expected_stop_time = (
            tracker.start_print_time
            + 1.5
            + (tracker.target_count_delta - stop_lead_counts - 125.0) / filtered_frequency
        )
        self.assertApproxEqual(tracker.stop_descriptor.print_time, expected_stop_time)

    def test_completed_move_rpm_scales_endpoint_after_rotation_distance_change(self):
        """Calibration changes must scale prior feed-forward by new commanded RPM."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc.set_rotation_distance(7.5)
        prime_tachometer_count(bldc)
        bldc.start_move(75.0, 600.0)
        first_tracker = bldc._finite_move_tracker
        count_time = first_tracker.start_print_time + 1.0
        bldc._schedule_finite_move_endpoint(count_time)
        settle_tracker_count(bldc, first_tracker, 120.0, count_time)
        bldc.get_last_move_revolutions(75.0)
        bldc.set_rotation_distance(15.0)
        bldc.start_move(75.0, 600.0)
        tracker = bldc._finite_move_tracker
        expected_stop_time = tracker.start_print_time + tracker.target_count_delta / 60.0
        self.assertApproxEqual(tracker.stop_descriptor.print_time, expected_stop_time)

    def test_completed_move_overshoot_increases_next_stop_lead(self):
        """Settled overrun must make following move cut PWM earlier without reverse braking."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc.set_rotation_distance(7.5)
        prime_tachometer_count(bldc)
        bldc.start_move(75.0, 600.0)
        first_tracker = bldc._finite_move_tracker
        final_count = first_tracker.target_count_delta + 5.0
        count_time = first_tracker.start_print_time + 1.0
        bldc._schedule_finite_move_endpoint(count_time)
        settle_tracker_count(bldc, first_tracker, final_count, count_time)
        bldc.get_last_move_revolutions(75.0)
        bldc.start_move(75.0, 600.0)
        self.assertApproxEqual(bldc._finite_move_tracker.stop_lead_counts, 3.5)

    def test_settled_rate_seed_does_not_apply_coast_lead_twice(self):
        """Settled effective rate already includes coast and must seed full target count."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc.set_rotation_distance(7.5)
        prime_tachometer_count(bldc)
        bldc.start_move(75.0, 600.0)
        first_tracker = bldc._finite_move_tracker
        final_count = first_tracker.target_count_delta + 5.0
        stop_time = first_tracker.start_print_time + 1.0
        bldc._schedule_finite_move_endpoint(stop_time)
        settle_tracker_count(bldc, first_tracker, final_count, stop_time)
        bldc.get_last_move_revolutions(75.0)
        bldc.start_move(75.0, 600.0)
        tracker = bldc._finite_move_tracker
        expected_stop_time = tracker.start_print_time + tracker.target_count_delta / final_count
        self.assertApproxEqual(tracker.stop_descriptor.print_time, expected_stop_time)

    def test_slow_finite_move_progress_reaches_count_endpoint(self):
        """A moving loaded motor must reach its count endpoint beyond the old fixed deadline."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc.set_rotation_distance(7.5)
        prime_tachometer_count(bldc)
        bldc.start_move(100.0, 600.0)
        tracker = bldc._finite_move_tracker
        first_sample_time = tracker.start_print_time + 1.0
        bldc.tachometer.handle_tachometer(first_sample_time, 100.0, first_sample_time)
        second_sample_time = tracker.start_print_time + 1.5
        bldc.tachometer.handle_tachometer(second_sample_time, 150.0, second_sample_time)
        expected_endpoint = tracker.start_print_time + tracker.get_pwm_off_target_counts() / 100.0
        self.assertApproxEqual(tracker.stop_descriptor.print_time, expected_endpoint)

    def test_stalled_finite_move_does_not_extend_stop_deadline(self):
        """Missing tach progress must retain a bounded terminal stop for motor safety."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        prime_tachometer_count(bldc)
        bldc.start_move(100.0, 600.0)
        tracker = bldc._finite_move_tracker
        initial_stop_time = tracker.stop_descriptor.print_time
        sample_time = tracker.start_print_time + 0.5
        bldc.tachometer.handle_tachometer(sample_time, 0.0, sample_time)
        self.assertApproxEqual(tracker.stop_descriptor.print_time, initial_stop_time)

    def test_finite_move_never_queues_reverse_drive_before_stop_1(self):
        """Stopping must not pull filament back after reaching a finite-move endpoint."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        distance = 100.0
        move_direction = 1
        bldc = bldc_runtime
        prime_tachometer_count(bldc)
        bldc.start_move(distance, 20.0)
        powered_directions = [
            descriptor.direction
            for descriptor, _source in bldc.motion_queue
            if descriptor.pwm is not None and descriptor.pwm > 0.0
        ]
        self.assertEqual(powered_directions, [move_direction] * len(powered_directions))

    def test_finite_move_never_queues_reverse_drive_before_stop_2(self):
        """Stopping must not pull filament back after reaching a finite-move endpoint."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        distance = -100.0
        move_direction = -1
        bldc = bldc_runtime
        prime_tachometer_count(bldc)
        bldc.start_move(distance, 20.0)
        powered_directions = [
            descriptor.direction
            for descriptor, _source in bldc.motion_queue
            if descriptor.pwm is not None and descriptor.pwm > 0.0
        ]
        self.assertEqual(powered_directions, [move_direction] * len(powered_directions))

    def test_finite_move_without_tach_sample_uses_timed_stop(self):
        """Startup before first tach sample must retain safe legacy finite-move behavior."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_runtime.start_move(100.0, 20.0)
        self.assertIs(bldc_runtime._finite_move_tracker, None)

    def test_brake_to_stop_runs_queued_brake_then_terminal_reset(self):
        """Sensor-triggered braking must end at PWM-off with stopped controller state."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        bldc.last_pwm = bldc.last_effective_pwm = 0.7
        bldc.last_dir = 1
        pwm_requests = capture_pwm_requests(bldc)
        bldc.brake_to_stop()
        bldc.reactor.pause(bldc.reactor.monotonic() + 1.0)
        self.assertApproxEqual(
            (bldc.motion_state, pwm_requests[-1][1]), (bldc.MOTION_STATE_STOP, 0.0)
        )

    def test_terminal_stop_wins_when_brake_window_ends(self):
        """Brake-window boundary must queue PWM-off without another reverse-drive tick."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc_module = self.bldc_module
        bldc = bldc_runtime
        current_print_time = active_print_time(bldc)
        brake_duration = 0.1
        bldc.motion_state = bldc.MOTION_STATE_BRAKE
        bldc.last_pwm = bldc.last_effective_pwm = 0.5
        bldc.motion_queue = [
            (
                bldc_module.MotionPwmDirect(
                    current_print_time - brake_duration, bldc.brake_pwm, brake_duration, -1
                ),
                "brake",
            ),
            (bldc_module.MotionStop(current_print_time), "brake"),
        ]
        pwm_requests = capture_pwm_requests(bldc)
        bldc._motion_timer_callback(bldc.reactor.monotonic())
        self.assertApproxEqual(
            (bldc.motion_state, pwm_requests[-1][1]), (bldc.MOTION_STATE_STOP, 0.0)
        )

    def test_last_move_revolutions_uses_tach_pulse_count(self):
        """Gear calibration must use observed motor revolutions instead of commanded runtime."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        prime_tachometer_count(bldc, count=100.0)
        fallback_wait = bldc.start_move(100.0, 20.0)
        settle_tracker_count(
            bldc,
            bldc._finite_move_tracker,
            1000.0,
            bldc._finite_move_tracker.start_print_time + 5.0,
        )
        bldc.wait_for_move(fallback_wait)
        revolutions = bldc.get_last_move_revolutions(100.0)
        self.assertApproxEqual(revolutions, 50.0)

    def test_tach_count_rotation_distance_uses_measured_filament(self):
        """BLDC gear calibration must divide measured filament by observed revolutions."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        prime_tachometer_count(bldc, count=100.0)
        fallback_wait = bldc.start_move(100.0, 20.0)
        settle_tracker_count(
            bldc,
            bldc._finite_move_tracker,
            1000.0,
            bldc._finite_move_tracker.start_print_time + 5.0,
        )
        bldc.wait_for_move(fallback_wait)
        rotation_distance = bldc.calculate_rotation_distance(90.0, 100.0)
        self.assertApproxEqual(rotation_distance, 1.8)

    def test_tach_count_rotation_distance_rejects_different_move_length(self):
        """A stale tach sample must not calibrate a differently sized requested move."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        prime_tachometer_count(bldc, count=100.0)
        fallback_wait = bldc.start_move(100.0, 20.0)
        settle_tracker_count(
            bldc,
            bldc._finite_move_tracker,
            1000.0,
            bldc._finite_move_tracker.start_print_time + 5.0,
        )
        bldc.wait_for_move(fallback_wait)
        rotation_distance = bldc.calculate_rotation_distance(90.0, 200.0)
        self.assertIs(rotation_distance, None)

    def test_tach_count_rotation_distance_rejects_active_move(self):
        """Calibration must not consume tach counts before finite motion completes."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        prime_tachometer_count(bldc)
        bldc.start_move(100.0, 20.0)
        bldc.tachometer.handle_tachometer(2.0, 180.0, 2.0)
        rotation_distance = bldc.calculate_rotation_distance(90.0, 100.0)
        self.assertIs(rotation_distance, None)

    def test_wait_for_finite_move_tracks_extended_count_guided_deadline(self):
        """Synchronous callers must not return before a tach-corrected late endpoint."""
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        prime_tachometer_count(bldc)
        fallback_wait = bldc.start_move(100.0, 20.0)
        sample_time = active_print_time(bldc) + 2.0
        bldc.tachometer.handle_tachometer(sample_time, 180.0, sample_time)
        expected_stop_time = bldc._finite_move_tracker.stop_descriptor.print_time
        bldc.wait_for_move(fallback_wait)
        self.assertGreaterEqual(bldc._get_scheduled_print_time(), expected_stop_time)

    def test_tachometer_caps_control_dt_after_stale_count_sample(self):
        bldc_runtime = _make_bldc_runtime(self.bldc_module)
        bldc = bldc_runtime
        tachometer = bldc.tachometer
        tachometer.handle_tachometer(1.0, 0.0, 1.0)
        tachometer.enabled = True
        tachometer.set_commanded(832.7, "move")
        tachometer.handle_tachometer(200.0, 1.0, 200.0)
        self.assertLess(tachometer.integral_correction_pwm, tachometer.control_max_delta_pwm)
        self.assertTrue(any(("control_dt=0.0800" in message for message in bldc.mmu.stepper_logs)))


if __name__ == "__main__":
    unittest.main()
