import os

import pytest

from test.support.bldc_fakes import (
    FakeConfig,
    FakeExtruder,
    FakeMmu,
    FakePrinter,
    FakeToolhead,
    load_mmu_gear_bldc_module,
)


@pytest.fixture()
def bldc_module():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    return load_mmu_gear_bldc_module(repo_root)


@pytest.fixture()
def bldc_runtime(bldc_module):
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


def capture_pwm_requests(bldc):
    requests = []

    def capture(mcu_pin, value, print_time):
        if mcu_pin is bldc.mcu_pwm_pin:
            requests.append((print_time, value))

    bldc._send_pin = capture
    return requests


def prepare_fresh_tachometer(bldc):
    bldc.tachometer.last_tach_eventtime = bldc._get_scheduled_print_time() + bldc.min_schedule_time


def active_print_time(bldc):
    return bldc._get_scheduled_print_time() + bldc.min_schedule_time


def test_motion_timer_keeps_active_descriptor_until_duration_expires(bldc_runtime, bldc_module):
    bldc = bldc_runtime
    bldc.motion_state = bldc.MOTION_STATE_MOVING
    descriptor = bldc_module.MotionDescriptor(active_print_time(bldc) - 0.01, 10.0, True, 1, duration=1.0)
    bldc.motion_queue = [(descriptor, "move")]
    capture_pwm_requests(bldc)

    bldc._motion_timer_callback(bldc.reactor.monotonic())
    assert bldc.motion_queue == [(descriptor, "move")]

    bldc.reactor.pause(2.0)
    bldc._motion_timer_callback(bldc.reactor.monotonic())
    assert bldc.motion_queue == []


def test_motion_timer_requeues_pwm_when_tach_correction_changes(bldc_runtime, bldc_module):
    bldc = bldc_runtime
    bldc.motion_state = bldc.MOTION_STATE_MOVING
    bldc.motion_queue = [
        (bldc_module.MotionDescriptor(active_print_time(bldc) - 0.01, 10.0, True, 1, duration=1.0), "move"),
    ]
    prepare_fresh_tachometer(bldc)
    bldc.tachometer.control_correction_pwm = 0.05
    pwm_requests = capture_pwm_requests(bldc)

    bldc._motion_timer_callback(bldc.reactor.monotonic())
    bldc.tachometer.control_correction_pwm = 0.09
    bldc.reactor.pause(0.02)
    bldc._motion_timer_callback(bldc.reactor.monotonic())

    assert [round(value, 4) for _, value in pwm_requests] == [0.2936, 0.3336]


def test_first_pid_enabled_dispatch_applies_existing_tach_correction(bldc_runtime, bldc_module):
    bldc = bldc_runtime
    bldc.motion_state = bldc.MOTION_STATE_MOVING
    bldc.motion_queue = [
        (bldc_module.MotionDescriptor(active_print_time(bldc) - 0.01, 10.0, True, 1, duration=1.0), "move"),
    ]
    prepare_fresh_tachometer(bldc)
    bldc.tachometer.enabled = False
    bldc.tachometer.control_correction_pwm = 0.05
    pwm_requests = capture_pwm_requests(bldc)

    bldc._motion_timer_callback(bldc.reactor.monotonic())

    assert pwm_requests[-1][1] == pytest.approx(0.293636, abs=0.000001)


def test_motion_timer_skips_small_pwm_correction_delta(bldc_runtime, bldc_module):
    bldc = bldc_runtime
    bldc.motion_state = bldc.MOTION_STATE_MOVING
    bldc.motion_queue = [
        (bldc_module.MotionDescriptor(active_print_time(bldc) - 0.01, 10.0, True, 1, duration=1.0), "move"),
    ]
    prepare_fresh_tachometer(bldc)
    bldc.tachometer.control_correction_pwm = 0.05
    pwm_requests = capture_pwm_requests(bldc)

    bldc._motion_timer_callback(bldc.reactor.monotonic())
    bldc.tachometer.control_correction_pwm = 0.051
    bldc.reactor.pause(0.02)
    bldc._motion_timer_callback(bldc.reactor.monotonic())

    assert len(pwm_requests) == 1


def test_motion_timer_stop_descriptor_sends_zero_and_stops_timer(bldc_runtime, bldc_module):
    bldc = bldc_runtime
    bldc.motion_state = bldc.MOTION_STATE_MOVING
    bldc.last_pwm = 0.5
    bldc.last_effective_pwm = 0.5
    bldc.motion_queue = [(bldc_module.MotionStop(active_print_time(bldc) - 0.01), "move")]
    pwm_requests = capture_pwm_requests(bldc)

    wake = bldc._motion_timer_callback(bldc.reactor.monotonic())

    assert wake == bldc.reactor.NEVER
    assert pwm_requests[-1][1] == pytest.approx(0.0)


def test_tachometer_caps_control_dt_after_stale_count_sample(bldc_runtime):
    bldc = bldc_runtime
    tachometer = bldc.tachometer
    tachometer.handle_tachometer(1.0, 0.0, 1.0)
    tachometer.enabled = True
    tachometer.set_commanded(832.7, "move")

    tachometer.handle_tachometer(200.0, 1.0, 200.0)

    assert tachometer.integral_correction_pwm < tachometer.control_max_delta_pwm
    assert any("control_dt=0.2000" in message for message in bldc.mmu.stepper_logs)
