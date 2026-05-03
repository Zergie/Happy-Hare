from __future__ import annotations

from pathlib import Path
from typing import Iterator
import sys

import pytest

TESTS_ROOT = Path(__file__).resolve().parent.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from run_test_rig_helpers import (
    assert_no_timer_too_close,
    assert_run_test_rig_healthy,
    backup_run_test_rig_startup_gcode,
    invoke_run_test_rig_scenario,
    restore_run_test_rig_startup_gcode,
)


@pytest.fixture(scope="module", autouse=True)
def restore_startup_gcode_after_tests() -> Iterator[None]:
    original_content = backup_run_test_rig_startup_gcode()
    try:
        yield
    finally:
        restore_run_test_rig_startup_gcode(original_content)


def test_calibrate_bldc_no_timer_too_close() -> None:
    result = invoke_run_test_rig_scenario(
        gcode_lines=["MMU_CALIBRATE_BLDC POINTS=3 SAVE=0"],
        expected_runtime_seconds=5.0,
    )

    assert_run_test_rig_healthy(result.log_text)
    assert_no_timer_too_close(result.log_text)
