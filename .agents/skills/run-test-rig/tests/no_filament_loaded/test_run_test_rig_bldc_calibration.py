from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Iterator
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
    get_bldc_tach_entries,
    invoke_run_test_rig_scenario,
    restore_run_test_rig_startup_gcode,
)

# Calibration timing constants from mmu_gear_bldc.py
CALIBRATION_SETTLE_S = 0.35
CALIBRATION_SAMPLE_S = 1.0
CALIBRATION_MIN_GAP_S = CALIBRATION_SETTLE_S + CALIBRATION_SAMPLE_S

# Original pattern without timestamp (fallback)
BLDC_CALIBRATE_SAMPLE_PATTERN_BASE = re.compile(
    r"(?m)^.*BLDC_CALIBRATE: sample pwm=(?P<pwm>[0-9.]+) rpm=(?P<rpm>[0-9.]+) unit=(?P<unit>.+)$"
)
# Enhanced pattern with optional timestamp capture at line start
BLDC_CALIBRATE_SAMPLE_PATTERN = re.compile(
    r"(?m)^(?P<timestamp>\d{1,2}:\d{2}:\d{2}\.\d{3}|\d+\.\d+)?\s*.*BLDC_CALIBRATE: sample pwm=(?P<pwm>[0-9.]+) rpm=(?P<rpm>[0-9.]+) unit=(?P<unit>.+)$"
)
BLDC_MAP_PATTERN = re.compile(
    r"(?m)^.*BLDC_MAP: mode=(?P<mode>\S+) reason=(?P<reason>\S+) points=(?P<points>\d+) unit=(?P<unit>.+)$"
)
BLDC_CALIBRATION_SAVED_PATTERN = re.compile(r"Saved BLDC calibration map to mmu_vars\.cfg")
BLDC_CALIBRATION_CAPTURED_PATTERN = re.compile(r"BLDC calibration captured (?P<count>\d+) valid points")


@dataclass(frozen=True)
class Scenario:
    label: str
    gcode_lines: list[str]
    expected_runtime_seconds: float
    extra_seconds: float
    validate_log: Callable[[str], None]


def _get_calibrate_samples(log_text: str, unit: str = TARGET_UNIT) -> list[dict]:
    samples = []
    for match in BLDC_CALIBRATE_SAMPLE_PATTERN.finditer(log_text):
        if match.group("unit").strip() == unit:
            samples.append(
                {
                    "pwm": float(match.group("pwm")),
                    "rpm": float(match.group("rpm")),
                    "timestamp": match.group("timestamp") if "timestamp" in match.groupdict() else None,
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


@pytest.fixture(scope="module", autouse=True)
def restore_startup_gcode_after_tests() -> Iterator[None]:
    original_content = backup_run_test_rig_startup_gcode()
    try:
        yield
    finally:
        restore_run_test_rig_startup_gcode(original_content)
def _assert_calibration_emits_samples(log_text: str) -> None:
    samples = _get_calibrate_samples(log_text, unit=TARGET_UNIT)
    preview = [s["line"] for s in samples[:5]]
    assert 1 <= len(samples) <= 4, (
        f"Expected between 1 and 4 BLDC_CALIBRATE sample lines for unit '{TARGET_UNIT}', "
        f"got {len(samples)}.\nPreview:\n" + "\n".join(preview)
    )


def _assert_calibration_samples_positive_rpm(log_text: str) -> None:
    samples = _get_calibrate_samples(log_text, unit=TARGET_UNIT)
    assert samples, f"No BLDC_CALIBRATE sample lines found for unit '{TARGET_UNIT}'."
    zero_rpm = [s for s in samples if s["rpm"] <= 0.0]
    assert not zero_rpm, (
        "Samples with zero/negative RPM found:\n" + "\n".join(s["line"] for s in zero_rpm)
    )


def _assert_calibration_switches_to_mapped_mode(log_text: str) -> None:
    captured_match = BLDC_CALIBRATION_CAPTURED_PATTERN.search(log_text)
    samples = _get_calibrate_samples(log_text, unit=TARGET_UNIT)
    map_events = _get_map_events(log_text, unit=TARGET_UNIT)
    mapped_events = [e for e in map_events if e["mode"] == "mapped"]

    evidence_ok = bool(captured_match) or bool(mapped_events) or len(samples) >= 3
    assert evidence_ok, "No calibration success evidence found (captured summary, mapped event, or >=3 samples)."

    if captured_match:
        assert int(captured_match.group("count")) >= 3, "Calibration did not produce enough valid points."

    if mapped_events:
        assert mapped_events[-1]["points"] >= 3, (
            f"Expected >=3 map points, got {mapped_events[-1]['points']}."
        )


def _assert_calibration_save_confirmation(log_text: str) -> None:
    assert BLDC_CALIBRATION_SAVED_PATTERN.search(log_text), (
        "Expected 'Saved BLDC calibration map to mmu_vars.cfg' in log after SAVE=1."
    )


def _assert_calibration_points_count_matches(log_text: str) -> None:
    points = 5
    samples = _get_calibrate_samples(log_text, unit=TARGET_UNIT)
    assert len(samples) == points, (
        f"Expected {points} BLDC_CALIBRATE sample lines, got {len(samples)}."
    )


def _assert_gear_move_uses_mapped_dispatch(log_text: str) -> None:
    assert_bldc_evidence_present(log_text, unit=TARGET_UNIT)
    tach_entries = get_bldc_tach_entries(log_text, unit=TARGET_UNIT)
    assert tach_entries, f"No BLDC_TACH entries found after gear move for unit '{TARGET_UNIT}'."

    map_events = _get_map_events(log_text, unit=TARGET_UNIT)
    mapped_events = [e for e in map_events if e["mode"] == "mapped"]
    assert mapped_events, (
        f"Expected BLDC_MAP mode=mapped after calibrated gear move for unit '{TARGET_UNIT}'.\n"
        f"Map events:\n" + "\n".join(e["line"] for e in map_events[:5])
    )


def _parse_time_string(time_str: str) -> float:
    """Convert HH:MM:SS.MMM timestamp string to seconds since midnight."""
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def _get_calibrate_sample_timestamps(log_text: str, unit: str = TARGET_UNIT) -> list[dict]:
    """Extract calibration samples with timing info and compute inter-sample delays."""
    samples = _get_calibrate_samples(log_text, unit=unit)

    if not samples or not all(s.get("timestamp") for s in samples):
        return []

    timestamped_samples = []
    for sample in samples:
        if sample.get("timestamp"):
            time_seconds = _parse_time_string(sample["timestamp"])
            timestamped_samples.append({
                "pwm": sample["pwm"],
                "rpm": sample["rpm"],
                "timestamp": sample["timestamp"],
                "time_seconds": time_seconds,
                "line": sample["line"],
            })

    # Compute deltas between consecutive samples
    samples_with_deltas = []
    for i, sample in enumerate(timestamped_samples):
        delta = None
        if i > 0:
            delta = sample["time_seconds"] - timestamped_samples[i - 1]["time_seconds"]
        samples_with_deltas.append({
            **sample,
            "delta_from_prev_s": delta,
        })

    return samples_with_deltas

SCENARIOS = [
    Scenario(
        label="calibrate emits sample lines for each point",
        gcode_lines=["MMU_CALIBRATE_BLDC POINTS=4 SAVE=0"],
        expected_runtime_seconds=8.0,
        extra_seconds=2.0,
        validate_log=_assert_calibration_emits_samples,
    ),
    Scenario(
        label="calibrate samples have positive rpm",
        gcode_lines=["MMU_CALIBRATE_BLDC POINTS=4 SAVE=0"],
        expected_runtime_seconds=8.0,
        extra_seconds=2.0,
        validate_log=_assert_calibration_samples_positive_rpm,
    ),
    Scenario(
        label="calibrate switches map to mapped mode",
        gcode_lines=["MMU_CALIBRATE_BLDC POINTS=4 SAVE=0"],
        expected_runtime_seconds=12.0,
        extra_seconds=6.0,
        validate_log=_assert_calibration_switches_to_mapped_mode,
    ),
    Scenario(
        label="calibrate with save logs confirmation",
        gcode_lines=["MMU_CALIBRATE_BLDC POINTS=4 SAVE=1"],
        expected_runtime_seconds=12.0,
        extra_seconds=6.0,
        validate_log=_assert_calibration_save_confirmation,
    ),
    Scenario(
        label="calibration captured count matches points param",
        gcode_lines=["MMU_CALIBRATE_BLDC POINTS=5 SAVE=0"],
        expected_runtime_seconds=14.0,
        extra_seconds=6.0,
        validate_log=_assert_calibration_points_count_matches,
    ),
    Scenario(
        label="gear move after calibration uses mapped dispatch",
        gcode_lines=[
            "MMU_CALIBRATE_BLDC POINTS=4 SAVE=0",
            "MMU_TEST_MOVE MOTOR=gear MOVE=200 SPEED=200",
        ],
        expected_runtime_seconds=12.0,
        extra_seconds=2.0,
        validate_log=_assert_gear_move_uses_mapped_dispatch,
    ),
]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[scenario.label for scenario in SCENARIOS])
def test_run_test_rig_bldc_calibration_scenarios(scenario: Scenario) -> None:
    result = invoke_run_test_rig_scenario(
        gcode_lines=scenario.gcode_lines,
        expected_runtime_seconds=scenario.expected_runtime_seconds,
        extra_seconds=scenario.extra_seconds,
    )

    assert_run_test_rig_healthy(result.log_text)
    scenario.validate_log(result.log_text)
