from __future__ import annotations

import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

RUN_TEST_RIG_ROOT = Path(__file__).resolve().parent.parent
STARTUP_GCODE_PATH = RUN_TEST_RIG_ROOT / "startup.gcode"
INVOKE_SCRIPT_PATH = RUN_TEST_RIG_ROOT / "invoke_test_rig.py"
EXPORT_SCRIPT_PATH = RUN_TEST_RIG_ROOT / "export_to_test_rig.py"
KLIPPY_LOG_PATH = RUN_TEST_RIG_ROOT / "klippy.log"
TARGET_UNIT = "mmu_gear_bldc unit0"

UNKNOWN_COMMAND_PATTERN = re.compile(r"Unknown command:")
TRACEBACK_HEADER_PATTERN = re.compile(r"Traceback \(most recent call last\)")
BENIGN_BLOCKING_IO_PATTERN = re.compile(r"BlockingIOError: \[Errno 11\] Resource temporarily unavailable")
BENIGN_RESPOND_RAW_PATTERN = re.compile(r"_respond_raw")
BLDC_TACH_PATTERN = re.compile(
    r"(?m)^.*BLDC_TACH: freq=(?P<freq>[0-9.]+) rpm=(?P<rpm>[0-9.]+)(?: time=(?P<time>[0-9.]+))?.* unit=(?P<unit>.+)$"
)
BLDC_PIN_PATTERN = re.compile(
    r"(?m)^.*BLDC_SET_PIN: (?P<message>.+?) applied=(?P<applied>[0-9.]+)(?: time=(?P<time>[0-9.]+))?.* unit=(?P<unit>.+)$"
)
BLDC_CONTROL_PATTERN = re.compile(
    r"(?m)^.*BLDC_CONTROL: source=(?P<source>\S+) reason=(?P<reason>\S+) "
    r"error_rpm=(?P<error_rpm>[-0-9.]+) base_pwm=(?P<base_pwm>[0-9.]+) "
    r"correction_pwm=(?P<correction_pwm>[-0-9.]+) integral_pwm=(?P<integral_pwm>[-0-9.]+) "
    r"applied_pwm=(?P<applied_pwm>[0-9.]+)(?: time=(?P<time>[0-9.]+))?.* unit=(?P<unit>.+)$"
)
BLDC_PREVIEW_PATTERN = re.compile(r"(?m)^.*BLDC_(?:TACH|SET_PIN).*$")


@dataclass(frozen=True)
class ScenarioResult:
    duration_seconds: int
    output: str
    log_text: str
    log_path: Path


@dataclass(frozen=True)
class BldcTachEntry:
    line: str
    frequency: float
    rpm: float
    unit: str


@dataclass(frozen=True)
class BldcPinEvent:
    line: str
    message: str
    applied: float
    time: float | None
    unit: str


@dataclass(frozen=True)
class RuntimeWindow:
    runtime_seconds: float
    start_event: BldcPinEvent
    stop_event: BldcPinEvent


@dataclass(frozen=True)
class BldcControlEntry:
    line: str
    source: str
    reason: str
    error_rpm: float
    base_pwm: float
    correction_pwm: float
    integral_pwm: float
    applied_pwm: float
    unit: str


def backup_run_test_rig_startup_gcode() -> str:
    if STARTUP_GCODE_PATH.exists():
        return STARTUP_GCODE_PATH.read_text(encoding="utf-8")
    return ""


def restore_run_test_rig_startup_gcode(content: str) -> None:
    STARTUP_GCODE_PATH.write_text(content, encoding="utf-8")


def set_run_test_rig_startup_gcode(lines: list[str]) -> None:
    STARTUP_GCODE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_run_test_rig_duration_seconds(
    expected_runtime_seconds: float,
    startup_seconds: float = 1.0,
    extra_seconds: float = 0.0,
    minimum_duration_seconds: int = 5,
) -> int:
    computed_duration = math.ceil((expected_runtime_seconds + startup_seconds + extra_seconds) * 1.5)
    return max(minimum_duration_seconds, computed_duration)


def invoke_run_test_rig_export() -> str:
    command = [sys.executable, str(EXPORT_SCRIPT_PATH)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode != 0:
        raise RuntimeError(f"Export failed. Output:\n{output}")
    return output


def invoke_run_test_rig_scenario(gcode_lines: list[str], expected_runtime_seconds: float, extra_seconds: float = 0.0) -> ScenarioResult:
    set_run_test_rig_startup_gcode(gcode_lines)
    duration_seconds = get_run_test_rig_duration_seconds(expected_runtime_seconds, extra_seconds=extra_seconds)
    command = [sys.executable, str(INVOKE_SCRIPT_PATH), "-DurationSeconds", str(duration_seconds)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = f"{result.stdout}\n{result.stderr}".strip()

    if result.returncode != 0:
        raise RuntimeError(f"Invoke failed. Output:\n{output}")

    if not KLIPPY_LOG_PATH.exists():
        raise RuntimeError(f"Missing klippy.log after invoke. Output:\n{output}")

    return ScenarioResult(
        duration_seconds=duration_seconds,
        output=output,
        log_text=KLIPPY_LOG_PATH.read_text(encoding="utf-8"),
        log_path=KLIPPY_LOG_PATH,
    )


def _is_benign_traceback(lines: list[str], traceback_index: int) -> bool:
    # Some rigs emit additional lines before the traceback tail; use a wider window.
    lookahead = lines[traceback_index : traceback_index + 24]
    has_blocking_io = any(BENIGN_BLOCKING_IO_PATTERN.search(line) for line in lookahead)
    has_respond_raw = any(BENIGN_RESPOND_RAW_PATTERN.search(line) for line in lookahead)
    return has_blocking_io and has_respond_raw


def get_run_test_rig_fatal_lines(log_text: str) -> list[str]:
    fatal_lines: list[str] = []
    lines = [line.rstrip() for line in log_text.splitlines()]

    for index, line in enumerate(lines):
        if UNKNOWN_COMMAND_PATTERN.search(line):
            fatal_lines.append(line.strip())
            continue

        if TRACEBACK_HEADER_PATTERN.search(line) and not _is_benign_traceback(lines, index):
            fatal_lines.append(line.strip())

    return fatal_lines


def assert_run_test_rig_healthy(log_text: str) -> None:
    fatal_lines = get_run_test_rig_fatal_lines(log_text)
    if fatal_lines:
        fatal_block = "\n".join(fatal_lines)
        raise AssertionError(f"Fatal run-test-rig lines found:\n{fatal_block}")


def get_bldc_tach_entries(log_text: str, unit: str = TARGET_UNIT) -> list[BldcTachEntry]:
    entries: list[BldcTachEntry] = []
    for match in BLDC_TACH_PATTERN.finditer(log_text):
        matched_unit = match.group("unit").strip()
        if matched_unit != unit:
            continue
        entries.append(
            BldcTachEntry(
                line=match.group(0).strip(),
                frequency=float(match.group("freq")),
                rpm=float(match.group("rpm")),
                unit=matched_unit,
            )
        )
    return entries


def get_bldc_pin_events(log_text: str, unit: str = TARGET_UNIT) -> list[BldcPinEvent]:
    events: list[BldcPinEvent] = []
    for match in BLDC_PIN_PATTERN.finditer(log_text):
        matched_unit = match.group("unit").strip()
        if matched_unit != unit:
            continue
        time_value = match.group("time")
        events.append(
            BldcPinEvent(
                line=match.group(0).strip(),
                message=match.group("message").strip(),
                applied=float(match.group("applied")),
                time=float(time_value) if time_value else None,
                unit=matched_unit,
            )
        )
    return events


def get_bldc_control_entries(log_text: str, unit: str = TARGET_UNIT) -> list[BldcControlEntry]:
    entries: list[BldcControlEntry] = []
    for match in BLDC_CONTROL_PATTERN.finditer(log_text):
        matched_unit = match.group("unit").strip()
        if matched_unit != unit:
            continue
        entries.append(
            BldcControlEntry(
                line=match.group(0).strip(),
                source=match.group("source").strip(),
                reason=match.group("reason").strip(),
                error_rpm=float(match.group("error_rpm")),
                base_pwm=float(match.group("base_pwm")),
                correction_pwm=float(match.group("correction_pwm")),
                integral_pwm=float(match.group("integral_pwm")),
                applied_pwm=float(match.group("applied_pwm")),
                unit=matched_unit,
            )
        )
    return entries


def get_bldc_runtime_seconds(log_text: str, unit: str = TARGET_UNIT) -> RuntimeWindow:
    timed_events = [
        event
        for event in get_bldc_pin_events(log_text, unit=unit)
        if event.time is not None and "PWM" in event.message
    ]
    start_event = next((event for event in timed_events if event.applied > 0.0), None)
    if start_event is None:
        raise RuntimeError(f"No non-zero BLDC_SET_PIN PWM event with time found for unit '{unit}'.")

    stop_event = next(
        (event for event in timed_events if event.time and event.time > start_event.time and event.applied <= 0.0),
        None,
    )
    if stop_event is None:
        raise RuntimeError(f"No zero BLDC_SET_PIN PWM event after first non-zero event found for unit '{unit}'.")

    return RuntimeWindow(
        runtime_seconds=float(stop_event.time - start_event.time),
        start_event=start_event,
        stop_event=stop_event,
    )


def get_bldc_observed_rpm(log_text: str, unit: str = TARGET_UNIT) -> float:
    entries = get_bldc_tach_entries(log_text, unit=unit)
    if not entries:
        raise RuntimeError(f"No BLDC_TACH lines found for unit '{unit}'.")
    return max(entry.rpm for entry in entries)


def get_bldc_evidence_preview(log_text: str, unit: str = TARGET_UNIT, count: int = 5) -> list[str]:
    preview: list[str] = []
    for match in BLDC_PREVIEW_PATTERN.finditer(log_text):
        line = match.group(0).strip()
        if unit in line:
            preview.append(line)
        if len(preview) >= count:
            break
    return preview


def assert_bldc_evidence_present(log_text: str, unit: str = TARGET_UNIT) -> None:
    tach_entries = get_bldc_tach_entries(log_text, unit=unit)
    pin_events = get_bldc_pin_events(log_text, unit=unit)
    if tach_entries and pin_events:
        return

    preview = get_bldc_evidence_preview(log_text, unit=unit)
    raise AssertionError(
        f"Missing BLDC evidence for unit '{unit}'. "
        f"TACH={len(tach_entries)} SET_PIN={len(pin_events)}. "
        f"Preview:\n{chr(10).join(preview)}"
    )


def assert_bldc_tach_control_pwm_raised_to_max(log_text: str, unit: str = TARGET_UNIT) -> None:
    control_entries = get_bldc_control_entries(log_text, unit=unit)
    saturated_control = [entry for entry in control_entries if entry.source == "active" and entry.applied_pwm >= 0.999]
    if not saturated_control:
        preview = [entry.line for entry in control_entries[:5]]
        raise AssertionError(
            f"No BLDC_CONTROL saturation found for unit '{unit}'. "
            f"Expected active control with applied_pwm=1.0000. "
            f"Preview:\n{chr(10).join(preview)}"
        )

    pin_events = get_bldc_pin_events(log_text, unit=unit)
    pwm_pin_events = [event for event in pin_events if "PWM" in event.message]
    saturated_pin = [event for event in pwm_pin_events if event.applied >= 0.999]
    if not saturated_pin:
        pin_preview = [event.line for event in pwm_pin_events[:5]]
        raise AssertionError(
            f"No BLDC_SET_PIN PWM saturation found for unit '{unit}'. "
            f"Expected applied=1.0000 on PWM pin. "
            f"Preview:\n{chr(10).join(pin_preview)}"
        )


def assert_bldc_tach_control_pwm_raised(log_text: str, unit: str = TARGET_UNIT) -> None:
    pin_events = get_bldc_pin_events(log_text, unit=unit)
    pwm_pin_events = [event for event in pin_events if "PWM" in event.message]
    if not pwm_pin_events:
        raise AssertionError(f"No BLDC_SET_PIN PWM events found for unit '{unit}'.")

    initial_pwm = pwm_pin_events[0].applied
    if initial_pwm >= 0.999:
        raise AssertionError(
            f"BLDC_SET_PIN PWM started saturated for unit '{unit}' (initial={initial_pwm:.4f}). "
            "Expected PWM to rise before reaching max."
        )

    later_pwm_values = [event.applied for event in pwm_pin_events[1:]]
    if not any(value > initial_pwm for value in later_pwm_values):
        pin_preview = [event.line for event in pwm_pin_events[:5]]
        raise AssertionError(
            f"BLDC_SET_PIN PWM never rose for unit '{unit}' (initial={initial_pwm:.4f}). "
            f"Preview:\n{chr(10).join(pin_preview)}"
        )


def assert_value_within_tolerance(label: str, observed: float, expected: float, tolerance: float, evidence_lines: list[str] | None = None) -> None:
    if abs(observed - expected) <= tolerance:
        return

    evidence_block = ""
    if evidence_lines:
        evidence_block = f"\nEvidence:\n{chr(10).join(evidence_lines)}"
    raise AssertionError(
        f"{label} mismatch. Expected {expected} +/- {tolerance} but observed {observed}.{evidence_block}"
    )
