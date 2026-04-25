from __future__ import annotations

import pytest

from run_test_rig_helpers import (
    assert_bldc_tach_control_pwm_raised,
    assert_bldc_tach_control_pwm_raised_to_max,
    get_bldc_control_entries,
    get_bldc_runtime_seconds,
    get_run_test_rig_fatal_lines,
)


def test_benign_blockingio_traceback_is_ignored() -> None:
    log_text = """
2026-04-22 12:00:00 Some line
Traceback (most recent call last):
  File \"/home/user/klipper/klippy/gcode.py\", line 471, in _respond_raw
    os.write(self.fd, (msg+\"\\n\").encode())
BlockingIOError: [Errno 11] Resource temporarily unavailable
2026-04-22 12:00:01 Continue run
"""
    assert get_run_test_rig_fatal_lines(log_text) == []


def test_unknown_traceback_stays_fatal() -> None:
    log_text = """
Traceback (most recent call last):
  File \"/home/user/klipper/klippy/other.py\", line 10, in fn
    raise RuntimeError(\"boom\")
RuntimeError: boom
"""
    fatal_lines = get_run_test_rig_fatal_lines(log_text)
    assert fatal_lines == ["Traceback (most recent call last):"]


def test_unknown_command_stays_fatal() -> None:
    log_text = """
MMU: startup
Unknown command: BAD_CMD
MMU: done
"""
    fatal_lines = get_run_test_rig_fatal_lines(log_text)
    assert fatal_lines == ["Unknown command: BAD_CMD"]


def test_get_bldc_control_entries_parses_fields() -> None:
    log_text = """
STEPPER: BLDC_CONTROL: source=active reason=active error_rpm=-4270.9 base_pwm=1.0000 correction_pwm=0.2127 integral_pwm=0.4262 applied_pwm=1.0000 unit=mmu_gear_bldc unit0
"""
    entries = get_bldc_control_entries(log_text)
    assert len(entries) == 1
    assert entries[0].source == "active"
    assert entries[0].applied_pwm == 1.0
    assert entries[0].error_rpm == -4270.9


def test_assert_bldc_tach_control_pwm_raised_to_max_passes_when_saturated() -> None:
    log_text = """
STEPPER: BLDC_CONTROL: source=active reason=active error_rpm=-1000.0 base_pwm=0.2000 correction_pwm=0.1000 integral_pwm=0.0000 applied_pwm=0.3000 unit=mmu_gear_bldc unit0
STEPPER: BLDC_CONTROL: source=active reason=active error_rpm=-4270.9 base_pwm=1.0000 correction_pwm=0.2127 integral_pwm=0.4262 applied_pwm=1.0000 unit=mmu_gear_bldc unit0
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=0.3000 time=170376.981 unit=mmu_gear_bldc unit0
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=1.0000 time=170376.991 unit=mmu_gear_bldc unit0
"""
    assert_bldc_tach_control_pwm_raised_to_max(log_text)


def test_assert_bldc_tach_control_pwm_raised_passes_when_pwm_rises() -> None:
    log_text = """
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=0.3000 time=170376.981 unit=mmu_gear_bldc unit0
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=1.0000 time=170376.991 unit=mmu_gear_bldc unit0
"""
    assert_bldc_tach_control_pwm_raised(log_text)


def test_assert_bldc_tach_control_pwm_raised_to_max_fails_without_saturation() -> None:
    log_text = """
STEPPER: BLDC_CONTROL: source=active reason=active error_rpm=-270.9 base_pwm=0.9000 correction_pwm=0.1127 integral_pwm=0.1262 applied_pwm=0.9500 unit=mmu_gear_bldc unit0
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=0.9500 time=170376.991 unit=mmu_gear_bldc unit0
"""
    with pytest.raises(AssertionError):
        assert_bldc_tach_control_pwm_raised_to_max(log_text)


def test_assert_bldc_tach_control_pwm_raised_to_max_fails_when_starting_at_max() -> None:
    log_text = """
STEPPER: BLDC_CONTROL: source=active reason=active error_rpm=-4270.9 base_pwm=1.0000 correction_pwm=0.2127 integral_pwm=0.4262 applied_pwm=1.0000 unit=mmu_gear_bldc unit0
STEPPER: BLDC_CONTROL: source=active reason=active error_rpm=-4200.0 base_pwm=1.0000 correction_pwm=0.2100 integral_pwm=0.4200 applied_pwm=1.0000 unit=mmu_gear_bldc unit0
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=1.0000 time=170376.981 unit=mmu_gear_bldc unit0
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=1.0000 time=170376.991 unit=mmu_gear_bldc unit0
"""
    with pytest.raises(AssertionError):
        assert_bldc_tach_control_pwm_raised(log_text)


def test_get_bldc_runtime_seconds_uses_pwm_events_only() -> None:
    log_text = """
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_DIR_0 applied=1.0000 time=100.000 unit=mmu_gear_bldc unit0
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=0.9000 time=101.500 unit=mmu_gear_bldc unit0
STEPPER: BLDC_SET_PIN: pin=YAMMU_BLDC_PWM_0 applied=0.0000 time=103.000 unit=mmu_gear_bldc unit0
"""
    runtime = get_bldc_runtime_seconds(log_text)
    assert runtime.runtime_seconds == pytest.approx(1.5)
