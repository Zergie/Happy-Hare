from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import sys

import pytest

TESTS_ROOT = Path(__file__).resolve().parent.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from run_test_rig_helpers import (
    assert_bldc_tach_control_pwm_raised,
    assert_bldc_tach_control_pwm_raised_to_max,
    get_bldc_control_entries,
    get_bldc_runtime_seconds,
    get_run_test_rig_fatal_lines,
)


@dataclass(frozen=True)
class FatalLinesScenario:
    label: str
    log_text: str
    expected_fatal_lines: list[str]


@dataclass(frozen=True)
class ControlAssertionScenario:
    label: str
    log_text: str
    assertion_fn: Callable[[str], None]
    should_raise: bool


FATAL_LINES_SCENARIOS = [
    FatalLinesScenario(
        label="benign blockingio traceback ignored",
        log_text="""
2026-04-22 12:00:00 Some line
Traceback (most recent call last):
  File \"/home/user/klipper/klippy/gcode.py\", line 471, in _respond_raw
    os.write(self.fd, (msg+\"\\n\").encode())
BlockingIOError: [Errno 11] Resource temporarily unavailable
2026-04-22 12:00:01 Continue run
""",
        expected_fatal_lines=[],
    ),
    FatalLinesScenario(
        label="unknown traceback stays fatal",
        log_text="""
Traceback (most recent call last):
  File \"/home/user/klipper/klippy/other.py\", line 10, in fn
    raise RuntimeError(\"boom\")
RuntimeError: boom
""",
        expected_fatal_lines=["Traceback (most recent call last):"],
    ),
    FatalLinesScenario(
        label="unknown command stays fatal",
        log_text="""
MMU: startup
Unknown command: BAD_CMD
MMU: done
""",
        expected_fatal_lines=["Unknown command: BAD_CMD"],
    ),
]


CONTROL_ASSERTION_SCENARIOS = [
    ControlAssertionScenario(
        label="tach control to max passes when saturated",
        log_text="""
STEPPER: BLDC_CONTROL: source=active reason=active error_rpm=-1000.0 base_pwm=0.2000 correction_pwm=0.1000 integral_pwm=0.0000 applied_pwm=0.3000 unit=mmu_gear_bldc unit0
STEPPER: BLDC_CONTROL: source=active reason=active error_rpm=-4270.9 base_pwm=1.0000 correction_pwm=0.2127 integral_pwm=0.4262 applied_pwm=1.0000 unit=mmu_gear_bldc unit0
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=0.3000 time=170376.981 unit=mmu_gear_bldc unit0
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=1.0000 time=170376.991 unit=mmu_gear_bldc unit0
""",
        assertion_fn=assert_bldc_tach_control_pwm_raised_to_max,
        should_raise=False,
    ),
    ControlAssertionScenario(
        label="tach control raised passes when pwm rises",
        log_text="""
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=0.3000 time=170376.981 unit=mmu_gear_bldc unit0
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=1.0000 time=170376.991 unit=mmu_gear_bldc unit0
""",
        assertion_fn=assert_bldc_tach_control_pwm_raised,
        should_raise=False,
    ),
    ControlAssertionScenario(
        label="tach control to max fails without saturation",
        log_text="""
STEPPER: BLDC_CONTROL: source=active reason=active error_rpm=-270.9 base_pwm=0.9000 correction_pwm=0.1127 integral_pwm=0.1262 applied_pwm=0.9500 unit=mmu_gear_bldc unit0
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=0.9500 time=170376.991 unit=mmu_gear_bldc unit0
""",
        assertion_fn=assert_bldc_tach_control_pwm_raised_to_max,
        should_raise=True,
    ),
    ControlAssertionScenario(
        label="tach control raised fails when starting at max",
        log_text="""
STEPPER: BLDC_CONTROL: source=active reason=active error_rpm=-4270.9 base_pwm=1.0000 correction_pwm=0.2127 integral_pwm=0.4262 applied_pwm=1.0000 unit=mmu_gear_bldc unit0
STEPPER: BLDC_CONTROL: source=active reason=active error_rpm=-4200.0 base_pwm=1.0000 correction_pwm=0.2100 integral_pwm=0.4200 applied_pwm=1.0000 unit=mmu_gear_bldc unit0
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=1.0000 time=170376.981 unit=mmu_gear_bldc unit0
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=1.0000 time=170376.991 unit=mmu_gear_bldc unit0
""",
        assertion_fn=assert_bldc_tach_control_pwm_raised,
        should_raise=True,
    ),
]


@pytest.mark.parametrize("scenario", FATAL_LINES_SCENARIOS, ids=[scenario.label for scenario in FATAL_LINES_SCENARIOS])
def test_fatal_line_detection_scenarios(scenario: FatalLinesScenario) -> None:
    fatal_lines = get_run_test_rig_fatal_lines(scenario.log_text)
    assert fatal_lines == scenario.expected_fatal_lines


def test_get_bldc_control_entries_parses_fields() -> None:
    log_text = """
STEPPER: BLDC_CONTROL: source=active reason=active error_rpm=-4270.9 base_pwm=1.0000 correction_pwm=0.2127 integral_pwm=0.4262 applied_pwm=1.0000 unit=mmu_gear_bldc unit0
"""
    entries = get_bldc_control_entries(log_text)
    assert len(entries) == 1
    assert entries[0].source == "active"
    assert entries[0].applied_pwm == 1.0
    assert entries[0].error_rpm == -4270.9


@pytest.mark.parametrize("scenario", CONTROL_ASSERTION_SCENARIOS, ids=[scenario.label for scenario in CONTROL_ASSERTION_SCENARIOS])
def test_bldc_control_assertion_scenarios(scenario: ControlAssertionScenario) -> None:
    if scenario.should_raise:
        with pytest.raises(AssertionError):
            scenario.assertion_fn(scenario.log_text)
        return
    scenario.assertion_fn(scenario.log_text)


def test_get_bldc_runtime_seconds_uses_pwm_events_only() -> None:
    log_text = """
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_DIR_0 applied=1.0000 time=100.000 unit=mmu_gear_bldc unit0
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=0.9000 time=101.500 unit=mmu_gear_bldc unit0
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=0.0000 time=103.000 unit=mmu_gear_bldc unit0
"""
    runtime = get_bldc_runtime_seconds(log_text)
    assert runtime.runtime_seconds == pytest.approx(1.5)
