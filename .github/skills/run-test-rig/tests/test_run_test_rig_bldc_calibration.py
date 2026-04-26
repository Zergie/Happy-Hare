from __future__ import annotations

import re

import pytest

from run_test_rig_helpers import (
    TARGET_UNIT,
    assert_bldc_evidence_present,
    assert_run_test_rig_healthy,
    backup_run_test_rig_startup_gcode,
    get_bldc_tach_entries,
    invoke_run_test_rig_scenario,
    restore_run_test_rig_startup_gcode,
)

# -- helpers ------------------------------------------------------------------

BLDC_CALIBRATE_SAMPLE_PATTERN = re.compile(
    r"(?m)^.*BLDC_CALIBRATE: sample pwm=(?P<pwm>[0-9.]+) rpm=(?P<rpm>[0-9.]+) unit=(?P<unit>.+)$"
)
BLDC_MAP_PATTERN = re.compile(
    r"(?m)^.*BLDC_MAP: mode=(?P<mode>\S+) reason=(?P<reason>\S+) points=(?P<points>\d+) unit=(?P<unit>.+)$"
)
BLDC_CALIBRATION_SAVED_PATTERN = re.compile(r"Saved BLDC calibration map to mmu_vars\.cfg")
BLDC_CALIBRATION_CAPTURED_PATTERN = re.compile(r"BLDC calibration captured (?P<count>\d+) valid points")


def _get_calibrate_samples(log_text: str, unit: str = TARGET_UNIT) -> list[dict]:
    samples = []
    for match in BLDC_CALIBRATE_SAMPLE_PATTERN.finditer(log_text):
        if match.group("unit").strip() == unit:
            samples.append(
                {
                    "pwm": float(match.group("pwm")),
                    "rpm": float(match.group("rpm")),
                    "line": match.group(0).strip(),
                }
            )
    return samples


def _get_map_events(log_text: str, unit: str = TARGET_UNIT) -> list[dict]:
    events = []
    for match in BLDC_MAP_PATTERN.finditer(log_text):
        if match.group("unit").strip() == unit:
            events.append(
                {
                    "mode": match.group("mode"),
                    "reason": match.group("reason"),
                    "points": int(match.group("points")),
                    "line": match.group(0).strip(),
                }
            )
    return events


# -- fixtures -----------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def restore_startup_gcode_after_tests() -> None:
    original_content = backup_run_test_rig_startup_gcode()
    try:
        yield
    finally:
        restore_run_test_rig_startup_gcode(original_content)


# -- tests --------------------------------------------------------------------


def test_calibrate_bldc_emits_sample_lines_for_each_point() -> None:
    """MMU_CALIBRATE_BLDC POINTS=4 should emit sample lines for valid tach points."""
    result = invoke_run_test_rig_scenario(
        gcode_lines=["MMU_CALIBRATE_BLDC POINTS=4 SAVE=0"],
        expected_runtime_seconds=8.0,
        extra_seconds=2.0,
    )

    assert_run_test_rig_healthy(result.log_text)
    samples = _get_calibrate_samples(result.log_text, unit=TARGET_UNIT)
    preview = [s["line"] for s in samples[:5]]
    assert 1 <= len(samples) <= 4, (
        f"Expected between 1 and 4 BLDC_CALIBRATE sample lines for unit '{TARGET_UNIT}', "
        f"got {len(samples)}.\nPreview:\n" + "\n".join(preview)
    )


def test_calibrate_bldc_samples_have_positive_rpm() -> None:
    """Every captured sample must report rpm > 0."""
    result = invoke_run_test_rig_scenario(
        gcode_lines=["MMU_CALIBRATE_BLDC POINTS=4 SAVE=0"],
        expected_runtime_seconds=8.0,
        extra_seconds=2.0,
    )

    assert_run_test_rig_healthy(result.log_text)
    samples = _get_calibrate_samples(result.log_text, unit=TARGET_UNIT)
    assert samples, f"No BLDC_CALIBRATE sample lines found for unit '{TARGET_UNIT}'."
    zero_rpm = [s for s in samples if s["rpm"] <= 0.0]
    assert not zero_rpm, (
        f"Samples with zero/negative RPM found:\n" + "\n".join(s["line"] for s in zero_rpm)
    )


def test_calibrate_bldc_switches_map_to_mapped_mode() -> None:
    """After MMU_CALIBRATE_BLDC, calibration should complete and mapped-mode evidence should appear when emitted."""
    result = invoke_run_test_rig_scenario(
        gcode_lines=["MMU_CALIBRATE_BLDC POINTS=4 SAVE=0"],
        expected_runtime_seconds=12.0,
        extra_seconds=6.0,
    )

    assert_run_test_rig_healthy(result.log_text)
    captured_match = BLDC_CALIBRATION_CAPTURED_PATTERN.search(result.log_text)
    samples = _get_calibrate_samples(result.log_text, unit=TARGET_UNIT)
    map_events = _get_map_events(result.log_text, unit=TARGET_UNIT)
    mapped_events = [e for e in map_events if e["mode"] == "mapped"]

    evidence_ok = bool(captured_match) or bool(mapped_events) or len(samples) >= 3
    assert evidence_ok, "No calibration success evidence found (captured summary, mapped event, or >=3 samples)."

    if captured_match:
        assert int(captured_match.group("count")) >= 3, "Calibration did not produce enough valid points."

    if mapped_events:
        assert mapped_events[-1]["points"] >= 3, (
            f"Expected >=3 map points, got {mapped_events[-1]['points']}."
        )


def test_calibrate_bldc_with_save_logs_save_confirmation() -> None:
    """MMU_CALIBRATE_BLDC SAVE=1 must log the save-confirmation line."""
    result = invoke_run_test_rig_scenario(
        gcode_lines=["MMU_CALIBRATE_BLDC POINTS=4 SAVE=1"],
        expected_runtime_seconds=12.0,
        extra_seconds=6.0,
    )

    assert_run_test_rig_healthy(result.log_text)
    assert BLDC_CALIBRATION_SAVED_PATTERN.search(result.log_text), (
        "Expected 'Saved BLDC calibration map to mmu_vars.cfg' in log after SAVE=1."
    )


def test_calibrate_bldc_captured_count_matches_points_param() -> None:
    """Calibration should emit one BLDC_CALIBRATE sample line per requested POINTS value."""
    points = 5
    result = invoke_run_test_rig_scenario(
        gcode_lines=[f"MMU_CALIBRATE_BLDC POINTS={points} SAVE=0"],
        expected_runtime_seconds=14.0,
        extra_seconds=6.0,
    )

    assert_run_test_rig_healthy(result.log_text)
    samples = _get_calibrate_samples(result.log_text, unit=TARGET_UNIT)
    assert len(samples) == points, (
        f"Expected {points} BLDC_CALIBRATE sample lines, got {len(samples)}."
    )


def test_gear_move_after_calibration_uses_mapped_dispatch() -> None:
    """After MMU_CALIBRATE_BLDC, a gear move must log BLDC_MAP mode=mapped (not linear)."""
    result = invoke_run_test_rig_scenario(
        gcode_lines=[
            "MMU_CALIBRATE_BLDC POINTS=4 SAVE=0",
            "MMU_TEST_MOVE MOTOR=gear MOVE=200 SPEED=200",
        ],
        expected_runtime_seconds=12.0,
        extra_seconds=2.0,
    )

    assert_run_test_rig_healthy(result.log_text)
    assert_bldc_evidence_present(result.log_text, unit=TARGET_UNIT)
    tach_entries = get_bldc_tach_entries(result.log_text, unit=TARGET_UNIT)
    assert tach_entries, f"No BLDC_TACH entries found after gear move for unit '{TARGET_UNIT}'."

    map_events = _get_map_events(result.log_text, unit=TARGET_UNIT)
    mapped_events = [e for e in map_events if e["mode"] == "mapped"]
    assert mapped_events, (
        f"Expected BLDC_MAP mode=mapped after calibrated gear move for unit '{TARGET_UNIT}'.\n"
        f"Map events:\n" + "\n".join(e["line"] for e in map_events[:5])
    )
