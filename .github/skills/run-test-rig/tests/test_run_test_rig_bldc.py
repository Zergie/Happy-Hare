from __future__ import annotations

from dataclasses import dataclass

import pytest

from run_test_rig_helpers import (
    TARGET_UNIT,
    assert_bldc_evidence_present,
    assert_bldc_tach_control_pwm_raised,
    assert_bldc_tach_control_pwm_raised_to_max,
    assert_run_test_rig_healthy,
    assert_value_within_tolerance,
    backup_run_test_rig_startup_gcode,
    get_bldc_observed_rpm,
    get_bldc_runtime_seconds,
    get_bldc_tach_entries,
    invoke_run_test_rig_scenario,
    restore_run_test_rig_startup_gcode,
)

RUNTIME_TOLERANCE_SECONDS = 0.11
RPM_TOLERANCE_RATIO = 0.10


@dataclass(frozen=True)
class Scenario:
    label: str
    gcode_lines: list[str]
    expected_runtime_seconds: float
    expected_bldc_runtime_seconds: float
    expected_rpm: float | None


SCENARIOS = [
    Scenario(
        label="sync gear motor follows extruder move and runs exactly 2.0s",
        gcode_lines=[
            "MMU_SYNC_GEAR_MOTOR SYNC=1",
            "_CLIENT_LINEAR_MOVE E=100 F=3000",
            "MMU_SYNC_GEAR_MOTOR SYNC=0",
        ],
        expected_runtime_seconds=4.0,
        expected_bldc_runtime_seconds=2.0,
        expected_rpm=None,
    ),
    Scenario(
        label="MMU_TEST_MOVE MOTOR=gear MOVE=400 SPEED=200",
        gcode_lines=["MMU_TEST_MOVE MOTOR=gear MOVE=400 SPEED=200"],
        expected_runtime_seconds=3.0,
        expected_bldc_runtime_seconds=2.0,
        expected_rpm=3000.0,
    ),
    Scenario(
        label="synced move shows BLDC evidence, 1.0s runtime, and 6000 rpm",
        gcode_lines=["MMU_TEST_MOVE MOTOR=synced MOVE=400 SPEED=200"],
        expected_runtime_seconds=2.0,
        expected_bldc_runtime_seconds=1.0,
        expected_rpm=6000.0,
    ),
]


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

    runtime = get_bldc_runtime_seconds(result.log_text, unit=TARGET_UNIT)
    assert_value_within_tolerance(
        label=f"{scenario.label} BLDC runtime",
        observed=runtime.runtime_seconds,
        expected=scenario.expected_bldc_runtime_seconds,
        tolerance=RUNTIME_TOLERANCE_SECONDS,
        evidence_lines=[runtime.start_event.line, runtime.stop_event.line],
    )

    if scenario.expected_rpm is None:
        return

    observed_rpm = get_bldc_observed_rpm(result.log_text, unit=TARGET_UNIT)
    rpm_tolerance = scenario.expected_rpm * RPM_TOLERANCE_RATIO
    tach_preview = [entry.line for entry in get_bldc_tach_entries(result.log_text, unit=TARGET_UNIT)[:5]]
    assert_value_within_tolerance(
        label=f"{scenario.label} BLDC rpm",
        observed=observed_rpm,
        expected=scenario.expected_rpm,
        tolerance=rpm_tolerance,
        evidence_lines=tach_preview,
    )


def test_blocked_bldc_tach_control_raises_pwm_to_max() -> None:
    result = invoke_run_test_rig_scenario(
        gcode_lines=["MMU_TEST_MOVE MOTOR=gear MOVE=400 SPEED=20"],
        expected_runtime_seconds=4.0,
    )

    assert_run_test_rig_healthy(result.log_text)
    assert_bldc_evidence_present(result.log_text, unit=TARGET_UNIT)
    assert_bldc_tach_control_pwm_raised(result.log_text, unit=TARGET_UNIT)
    assert_bldc_tach_control_pwm_raised_to_max(result.log_text, unit=TARGET_UNIT)
