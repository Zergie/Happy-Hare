from __future__ import annotations

from dataclasses import dataclass

import pytest

from run_test_rig_helpers import (
    TARGET_UNIT,
    assert_bldc_evidence_present,
    assert_bldc_tach_control_pwm_raised,
    assert_run_test_rig_healthy,
    backup_run_test_rig_startup_gcode,
    get_bldc_runtime_seconds,
    get_bldc_tach_entries,
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
        gcode_lines=["MMU_TEST_MOVE MOTOR=gear MOVE=400 SPEED=20"],
        expected_runtime_seconds=8.0,
    )

    assert_run_test_rig_healthy(result.log_text)
    assert_bldc_evidence_present(result.log_text, unit=TARGET_UNIT)
    assert_bldc_tach_control_pwm_raised(result.log_text, unit=TARGET_UNIT)
