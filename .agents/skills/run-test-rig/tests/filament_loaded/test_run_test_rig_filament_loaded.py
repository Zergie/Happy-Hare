from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import sys

import pytest

TESTS_ROOT = Path(__file__).resolve().parent.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from run_test_rig_helpers import (
    assert_run_test_rig_healthy,
    backup_run_test_rig_startup_gcode,
    close_run_test_rig_session,
    invoke_run_test_rig_scenario,
    query_run_test_rig_filament_sensor,
    restore_run_test_rig_startup_gcode,
    set_run_test_rig_session_reuse,
)

@dataclass(frozen=True)
class Scenario:
    label: str
    gcode_lines: list[str]
    expected_runtime_seconds: float


SCENARIOS = [
    Scenario(
        label="E=50 — 4 synced moves",
        gcode_lines=[
            "MMU_SYNC_GEAR_MOTOR SYNC=1",
            "_CLIENT_LINEAR_MOVE E=50 F=3000",
            "_CLIENT_LINEAR_MOVE E=-50 F=3000",
            "_CLIENT_LINEAR_MOVE E=50 F=3000",
            "_CLIENT_LINEAR_MOVE E=-50 F=3000",
            "MMU_SYNC_GEAR_MOTOR SYNC=0",
        ],
        expected_runtime_seconds=15.0,
    ),
    Scenario(
        label="E=10 — 4 synced moves",
        gcode_lines=[
            "MMU_SYNC_GEAR_MOTOR SYNC=1",
            "_CLIENT_LINEAR_MOVE E=10 F=3000",
            "_CLIENT_LINEAR_MOVE E=-10 F=3000",
            "_CLIENT_LINEAR_MOVE E=10 F=3000",
            "_CLIENT_LINEAR_MOVE E=-10 F=3000",
            "MMU_SYNC_GEAR_MOTOR SYNC=0",
        ],
        expected_runtime_seconds=5.0,
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


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.label for s in SCENARIOS])
def test_sync_gear_extruder_moves_no_lost_steps(scenario: Scenario) -> None:
    result = invoke_run_test_rig_scenario(
        gcode_lines=scenario.gcode_lines,
        expected_runtime_seconds=scenario.expected_runtime_seconds,
    )

    assert_run_test_rig_healthy(result.log_text)

    assert query_run_test_rig_filament_sensor("extruder_sensor"), "extruder_sensor not triggered — is filament loaded?"
    assert query_run_test_rig_filament_sensor("toolhead_sensor"), "toolhead_sensor not triggered — is filament loaded?"

    answer = input("\n[%s] Observe the extruder. Did it lose any steps during the 4 moves? (y/N): " % scenario.label)
    if answer.strip().lower() in ("y", "yes"):
        pytest.fail("Operator reported lost steps")
