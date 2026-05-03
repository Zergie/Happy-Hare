from __future__ import annotations

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


def test_mmu_check_gate_with_filament_loaded() -> None:
    assert query_run_test_rig_filament_sensor("extruder_sensor"), \
        "extruder_sensor not triggered — is filament loaded?"
    assert query_run_test_rig_filament_sensor("toolhead_sensor"), \
        "toolhead_sensor not triggered — is filament loaded?"

    result = invoke_run_test_rig_scenario(
        gcode_lines=["MMU_CHECK_GATE"],
        expected_runtime_seconds=5.0,
    )

    assert_run_test_rig_healthy(result.log_text)

    assert not query_run_test_rig_filament_sensor("unit_0_mmu_gate_sensor"), \
        "unit_0_mmu_gate_sensor triggered after MMU_CHECK_GATE — unexpected filament detected at gate"
