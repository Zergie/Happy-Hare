from __future__ import annotations

import atexit
import importlib.util
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
TIMER_TOO_CLOSE_PATTERN = re.compile(r"Timer too close")
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


@dataclass
class RunTestRigSession:
    invoke_module: Any
    process: subprocess.Popen[str]
    client: Any
    log_offset: int = 0


SESSION_REUSE_DISABLED = "disabled"
SESSION_REUSE_ENABLED = "enabled"

_session_reuse_mode = SESSION_REUSE_DISABLED
_active_session: RunTestRigSession | None = None



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


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module '{module_name}' from path: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_invoke_module_from_path():
    return _load_module_from_path("run_test_rig_invoke", INVOKE_SCRIPT_PATH)


def _load_moonraker_module_from_path():
    moonraker_path = RUN_TEST_RIG_ROOT / "moonraker_client.py"
    return _load_module_from_path("run_test_rig_moonraker_client", moonraker_path)


def set_run_test_rig_session_reuse(enabled: bool) -> None:
    global _session_reuse_mode
    _session_reuse_mode = SESSION_REUSE_ENABLED if enabled else SESSION_REUSE_DISABLED
    if not enabled:
        close_run_test_rig_session()


def _ensure_active_run_test_rig_session() -> RunTestRigSession:
    global _active_session
    if _active_session is not None:
        return _active_session

    invoke_module = _load_invoke_module_from_path()
    moonraker_module = _load_moonraker_module_from_path()

    invoke_module.ensure_no_running_klippy()
    invoke_module.remove_remote_klippy_log()
    process = invoke_module.start_remote_klippy_session()

    client = moonraker_module.MoonrakerClient(invoke_module.MOONRAKER_URL)
    client.ensure_ready()

    _active_session = RunTestRigSession(
        invoke_module=invoke_module,
        process=process,
        client=client,
        log_offset=0,
    )
    return _active_session


def close_run_test_rig_session() -> None:
    global _active_session
    if _active_session is None:
        return

    session = _active_session
    _active_session = None

    try:
        session.invoke_module.ensure_no_running_klippy()
    finally:
        session.invoke_module.terminate_process(session.process)


def _slice_log_since_last_offset(full_log_text: str, session: RunTestRigSession) -> str:
    if session.log_offset <= 0:
        session.log_offset = len(full_log_text)
        return full_log_text

    if session.log_offset > len(full_log_text):
        session.log_offset = len(full_log_text)
        return full_log_text

    sliced = full_log_text[session.log_offset :]
    session.log_offset = len(full_log_text)
    return sliced if sliced.strip() else full_log_text


def _invoke_run_test_rig_scenario_reuse_session(gcode_lines: list[str], duration_seconds: int) -> ScenarioResult:
    session = _ensure_active_run_test_rig_session()

    session.invoke_module.run_gcode_via_moonraker(gcode_lines, duration_seconds, session.client)
    full_log_text = session.client.get_klippy_log()
    KLIPPY_LOG_PATH.write_text(full_log_text, encoding="utf-8")
    scenario_log_text = _slice_log_since_last_offset(full_log_text, session)

    return ScenarioResult(
        duration_seconds=duration_seconds,
        output="invoke_test_rig.reuse_session",
        log_text=scenario_log_text,
        log_path=KLIPPY_LOG_PATH,
    )


def invoke_run_test_rig_scenario(gcode_lines: list[str], expected_runtime_seconds: float, extra_seconds: float = 0.0) -> ScenarioResult:
    set_run_test_rig_startup_gcode(gcode_lines)
    duration_seconds = get_run_test_rig_duration_seconds(expected_runtime_seconds, extra_seconds=extra_seconds)

    if _session_reuse_mode == SESSION_REUSE_ENABLED:
        return _invoke_run_test_rig_scenario_reuse_session(gcode_lines, duration_seconds)

    invoke_module = _load_invoke_module_from_path()
    run_scenario = getattr(invoke_module, "run_scenario", None)
    if run_scenario is None:
        raise RuntimeError("Invoke module is missing required function 'run_scenario'.")

    log_path = run_scenario(gcode_lines=gcode_lines, duration_seconds=duration_seconds)

    resolved_log_path = Path(log_path) if log_path else KLIPPY_LOG_PATH
    output = "invoke_test_rig.run_scenario"
    if not resolved_log_path.exists():
        raise RuntimeError(f"Missing klippy.log after invoke. Output:\n{output}")

    return ScenarioResult(
        duration_seconds=duration_seconds,
        output=output,
        log_text=resolved_log_path.read_text(encoding="utf-8"),
        log_path=resolved_log_path,
    )


atexit.register(close_run_test_rig_session)


def query_run_test_rig_filament_sensor(sensor_name: str) -> bool:
    if _session_reuse_mode != SESSION_REUSE_ENABLED:
        raise RuntimeError("query_run_test_rig_filament_sensor requires session reuse to be enabled")
    session = _ensure_active_run_test_rig_session()
    key = "filament_switch_sensor %s" % sensor_name
    response = session.client.query_objects({key: None})
    return bool(response["result"]["status"][key]["filament_detected"])


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


def assert_no_timer_too_close(log_text: str) -> None:
    matches = TIMER_TOO_CLOSE_PATTERN.findall(log_text)
    if matches:
        raise AssertionError(
            "'Timer too close' error reported in klippy.log. "
            "Scheduling conflict during command execution."
        )


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


def _is_pwm_pin_event(event: BldcPinEvent) -> bool:
    msg = event.message
    return "PWM" in msg or msg.startswith("kick ") or msg.startswith("discard ")


def get_bldc_runtime_seconds(log_text: str, unit: str = TARGET_UNIT) -> RuntimeWindow:
    timed_events = [
        event
        for event in get_bldc_pin_events(log_text, unit=unit)
        if event.time is not None and _is_pwm_pin_event(event)
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
    pwm_pin_events = [event for event in pin_events if _is_pwm_pin_event(event)]
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
    pwm_pin_events = [event for event in pin_events if _is_pwm_pin_event(event)]
    if not pwm_pin_events:
        raise AssertionError(f"No BLDC_SET_PIN PWM events found for unit '{unit}'.")

    # Use actual PWM pin writes as baseline and ignore kick/discard pseudo-events.
    baseline_events = [event for event in pwm_pin_events if event.message.startswith("pin=")]
    if baseline_events:
        pwm_pin_events = baseline_events

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


