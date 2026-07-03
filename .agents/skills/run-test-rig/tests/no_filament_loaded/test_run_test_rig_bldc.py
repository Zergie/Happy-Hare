from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import re
import sys

import pytest

TESTS_ROOT = Path(__file__).resolve().parent.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from run_test_rig_helpers import (
    TARGET_UNIT,
    assert_bldc_evidence_present,
    assert_run_test_rig_healthy,
    backup_run_test_rig_startup_gcode,
    close_run_test_rig_session,
    get_bldc_runtime_seconds,
    get_bldc_tach_entries,
    invoke_run_test_rig_scenario,
    restore_run_test_rig_startup_gcode,
    set_run_test_rig_session_reuse,
)

RUNTIME_TOLERANCE_SECONDS = 0.11
BLDC_STALE_SAMPLE_PATTERN = re.compile(r"(?m)^.*BLDC_PROCESS_MOVE: stale sample(?:\s|$).*$")


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
        expected_runtime_seconds=2.0,
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
        label="MMU_TEST_MOVE MOTOR=gear MOVE=400 SPEED=200 WAIT=0 reaches significant RPM",
        gcode_lines=["MMU_TEST_MOVE MOTOR=gear MOVE=400 SPEED=200 WAIT=0"],
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
    Scenario(
        label="high-then-low gear speed completes without TTC",
        gcode_lines=[
            "MMU_TEST_MOVE MOTOR=gear MOVE=400 SPEED=300",
            "MMU_TEST_MOVE MOTOR=gear MOVE=50 SPEED=15",
        ],
        expected_runtime_seconds=30.0,
        expected_bldc_runtime_seconds=None,
        expected_min_rpm=None,
    ),
]

@pytest.fixture(scope="module", autouse=True)
def restore_startup_gcode_after_tests() -> Iterator[None]:
    original_content = backup_run_test_rig_startup_gcode()
    try:
        yield
    finally:
        restore_run_test_rig_startup_gcode(original_content)


@pytest.fixture(scope="module", autouse=True)
def reuse_klippy_session_for_module() -> Iterator[None]:
    set_run_test_rig_session_reuse(True)
    try:
        yield
    finally:
        close_run_test_rig_session()
        set_run_test_rig_session_reuse(False)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[scenario.label for scenario in SCENARIOS])
def test_run_test_rig_bldc_scenarios(scenario: Scenario) -> None:
    result = invoke_run_test_rig_scenario(
        gcode_lines=scenario.gcode_lines,
        expected_runtime_seconds=scenario.expected_runtime_seconds,
    )

    assert_run_test_rig_healthy(result.log_text)
    assert_bldc_evidence_present(result.log_text, unit=TARGET_UNIT)

    if scenario.label == "sync gear motor follows extruder move":
        stale_sample_lines = BLDC_STALE_SAMPLE_PATTERN.findall(result.log_text)
        assert not stale_sample_lines, (
            "BLDC late-start evidence detected (stale sample) during synced extruder move.\n"
            "Evidence:\n" + "\n".join(stale_sample_lines[:8])
        )

    if scenario.label == "high-then-low gear speed completes without TTC":
        assert re.search(r"BLDC_CONTROL:.*reason=pid_disabled", result.log_text), (
            "Expected BLDC PID to be disabled during kick phase at low-speed transition, but no evidence found."
        )

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
