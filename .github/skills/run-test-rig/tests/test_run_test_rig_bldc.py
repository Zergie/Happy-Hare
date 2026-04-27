from __future__ import annotations

from dataclasses import dataclass

import pytest

from run_test_rig_helpers import (
    TARGET_UNIT,
    assert_bldc_evidence_present,
    assert_bldc_pulse_edges_and_clean_stop,
    assert_bldc_pulsed_mode_present,
    assert_bldc_starts_pulsed_and_ends_active_at_max_pwm,
    assert_bldc_tach_control_pwm_raised,
    assert_run_test_rig_healthy,
    backup_run_test_rig_startup_gcode,
    get_bldc_runtime_seconds,
    get_bldc_tach_entries,
    get_bldc_terminal_runtime_seconds,
    invoke_run_test_rig_export,
    invoke_run_test_rig_scenario,
    restore_run_test_rig_startup_gcode,
)

RUNTIME_TOLERANCE_SECONDS = 0.11


@dataclass(frozen=True)
class Scenario:
    label: str
    gcode_lines: list[str]
    expected_runtime_seconds: float
    expected_bldc_runtime_seconds: float | None
    expected_min_rpm: float | None


SCENARIOS = [
    Scenario(
        label="sync gear motor follows extruder move",
        gcode_lines=[
            "MMU_SYNC_GEAR_MOTOR SYNC=1",
            "_CLIENT_LINEAR_MOVE E=100 F=3000",
            "MMU_SYNC_GEAR_MOTOR SYNC=0",
        ],
        expected_runtime_seconds=7.0,
        expected_bldc_runtime_seconds=None,
        expected_min_rpm=None,
    ),
    Scenario(
        label="MMU_TEST_MOVE MOTOR=gear MOVE=400 SPEED=200 reaches significant RPM",
        gcode_lines=["MMU_TEST_MOVE MOTOR=gear MOVE=400 SPEED=200"],
        expected_runtime_seconds=12.0,
        expected_bldc_runtime_seconds=None,
        expected_min_rpm=2000.0,
    ),
    Scenario(
        label="synced move shows BLDC evidence and significant RPM",
        gcode_lines=["MMU_TEST_MOVE MOTOR=synced MOVE=400 SPEED=200"],
        expected_runtime_seconds=14.0,
        expected_bldc_runtime_seconds=None,
        expected_min_rpm=2000.0,
    ),
]


@pytest.fixture(scope="module", autouse=True)
def export_code_to_test_rig_once() -> None:
    invoke_run_test_rig_export()


@pytest.fixture(scope="module", autouse=True)
def restore_startup_gcode_after_tests() -> None:
    original_content = backup_run_test_rig_startup_gcode()
    try:
        yield
    finally:
        restore_run_test_rig_startup_gcode(original_content)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[scenario.label for scenario in SCENARIOS])
def test_run_test_rig_bldc_scenarios(scenario: Scenario) -> None:
    result = invoke_run_test_rig_scenario(
        gcode_lines=scenario.gcode_lines,
        expected_runtime_seconds=scenario.expected_runtime_seconds,
    )

    assert_run_test_rig_healthy(result.log_text)
    assert_bldc_evidence_present(result.log_text, unit=TARGET_UNIT)

    if scenario.expected_bldc_runtime_seconds is not None:
        runtime = get_bldc_runtime_seconds(result.log_text, unit=TARGET_UNIT)

        if abs(runtime.runtime_seconds - scenario.expected_bldc_runtime_seconds) > RUNTIME_TOLERANCE_SECONDS:
            evidence = [runtime.start_event.line, runtime.stop_event.line]
            raise AssertionError(
                f"{scenario.label} BLDC runtime mismatch. "
                f"Expected {scenario.expected_bldc_runtime_seconds} +/- {RUNTIME_TOLERANCE_SECONDS} "
                f"but observed {runtime.runtime_seconds}.\nEvidence:\n" + "\n".join(evidence)
            )
    if scenario.expected_min_rpm is None:
        return

    tach_entries = get_bldc_tach_entries(result.log_text, unit=TARGET_UNIT)
    observed_max_rpm = max((entry.rpm for entry in tach_entries), default=0.0)
    tach_preview = [entry.line for entry in tach_entries[:5]]
    assert observed_max_rpm >= scenario.expected_min_rpm, (
        f"{scenario.label} BLDC peak RPM too low. "
        f"Expected >= {scenario.expected_min_rpm} but observed {observed_max_rpm}.\n"
        f"Evidence:\n" + "\n".join(tach_preview)
    )


def test_blocked_bldc_tach_control_raises_pwm_to_max() -> None:
    result = invoke_run_test_rig_scenario(
        gcode_lines=["MMU_TEST_MOVE MOTOR=gear MOVE=400 SPEED=40"],
        expected_runtime_seconds=8.0,
    )

    assert_run_test_rig_healthy(result.log_text)
    assert_bldc_evidence_present(result.log_text, unit=TARGET_UNIT)
    assert_bldc_tach_control_pwm_raised(result.log_text, unit=TARGET_UNIT)


def test_blocked_bldc_pulsed_move_transitions_to_active_mode_at_max_pwm() -> None:
    result = invoke_run_test_rig_scenario(
        gcode_lines=[
            "MMU_TEST_MOVE MOTOR=gear MOVE=20 SPEED=2",
            "MMU_TEST_MOVE MOTOR=gear MOVE=400 SPEED=40",
        ],
        expected_runtime_seconds=8.0,
    )

    assert_run_test_rig_healthy(result.log_text)
    assert_bldc_evidence_present(result.log_text, unit=TARGET_UNIT)
    assert_bldc_starts_pulsed_and_ends_active_at_max_pwm(result.log_text, unit=TARGET_UNIT)


def test_low_speed_gear_move_uses_pulsed_control_and_stops_cleanly() -> None:
    # Very low speed to force commanded RPM below pulsed fallback entry threshold.
    move_mm = 20.0
    speed_mm_s = 2.0
    result = invoke_run_test_rig_scenario(
        gcode_lines=[f"MMU_TEST_MOVE MOTOR=gear MOVE={move_mm} SPEED={speed_mm_s}"],
        expected_runtime_seconds=14.0,
    )

    assert_run_test_rig_healthy(result.log_text)
    assert_bldc_evidence_present(result.log_text, unit=TARGET_UNIT)
    assert_bldc_pulsed_mode_present(result.log_text, unit=TARGET_UNIT)
    assert_bldc_pulse_edges_and_clean_stop(result.log_text, unit=TARGET_UNIT)


def test_pulsed_move_runtime_scales_with_length() -> None:
    speed_mm_s = 2.0
    short_move_mm = 10.0
    long_move_mm = 20.0

    short_result = invoke_run_test_rig_scenario(
        gcode_lines=[f"MMU_TEST_MOVE MOTOR=gear MOVE={short_move_mm} SPEED={speed_mm_s}"],
        expected_runtime_seconds=10.0,
    )
    long_result = invoke_run_test_rig_scenario(
        gcode_lines=[f"MMU_TEST_MOVE MOTOR=gear MOVE={long_move_mm} SPEED={speed_mm_s}"],
        expected_runtime_seconds=14.0,
    )

    assert_run_test_rig_healthy(short_result.log_text)
    assert_run_test_rig_healthy(long_result.log_text)
    assert_bldc_pulsed_mode_present(short_result.log_text, unit=TARGET_UNIT)
    assert_bldc_pulsed_mode_present(long_result.log_text, unit=TARGET_UNIT)

    short_runtime = get_bldc_terminal_runtime_seconds(short_result.log_text, unit=TARGET_UNIT).runtime_seconds
    long_runtime = get_bldc_terminal_runtime_seconds(long_result.log_text, unit=TARGET_UNIT).runtime_seconds
    observed_ratio = long_runtime / short_runtime if short_runtime > 0 else 0.0
    expected_ratio = long_move_mm / short_move_mm

    assert abs(observed_ratio - expected_ratio) <= 0.35, (
        f"Pulsed runtime-length scaling mismatch. Expected ratio {expected_ratio:.2f} "
        f"for MOVE {short_move_mm}->{long_move_mm}, observed {observed_ratio:.2f} "
        f"(short={short_runtime:.3f}s long={long_runtime:.3f}s)."
    )
