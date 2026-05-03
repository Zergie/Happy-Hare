from __future__ import annotations

import argparse
import subprocess
import time
import sys
import traceback
from pathlib import Path


class KlippyFailureError(RuntimeError):
    """RuntimeError with attached klippy.log error context."""
    def __init__(self, message: str, error_context: str = "") -> None:
        self.error_context = error_context
        super().__init__(message + error_context)

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

KLIPPER_HOST = "klippy-test"
MOONRAKER_URL = f"http://{KLIPPER_HOST}:7125"
KLIPPY_LOG = "/home/user/printer_data/logs/klippy.log"
KLIPPY_PROCESS = "/home/user/klipper/klippy/klippy.py"
KLIPPY_COMMAND = (
    f"/home/user/klippy-env/bin/python {KLIPPY_PROCESS} "
    "/home/user/printer_data/config/printer.cfg "
    "-I /home/user/printer_data/comms/klippy.serial "
    f"-l {KLIPPY_LOG} "
    "-a /home/user/printer_data/comms/klippy.sock;"
)


def script_root() -> Path:
    return SCRIPT_ROOT


def startup_gcode_path() -> Path:
    return script_root() / "startup.gcode"


def local_klippy_log_path() -> Path:
    return script_root() / "klippy.log"


def run_ssh(command: str, check: bool = True, timeout_seconds: float = 30.0) -> str:
    full_command = ["ssh", KLIPPER_HOST, command]
    print(" ".join(full_command))
    result = subprocess.run(full_command, capture_output=True, text=True, check=False, timeout=timeout_seconds)
    if check and result.returncode != 0:
        raise RuntimeError(f"SSH command failed: {command}\n{result.stderr}")
    return result.stdout

def read_gcode_lines() -> list[str]:
    content = startup_gcode_path().read_text(encoding="utf-8")
    return [line.strip() for line in content.splitlines() if line.strip()]


def ensure_no_running_klippy() -> None:
    # Kill only klippy.py process, not arbitrary python helpers running on rig.
    run_ssh("pgrep -f '%s' && kill `pgrep -f '%s'`" % (KLIPPY_PROCESS, KLIPPY_PROCESS), check=False)


def remove_remote_klippy_log() -> None:
    run_ssh(f"rm {KLIPPY_LOG}", check=False)


def start_remote_klippy_session() -> subprocess.Popen[str]:
    command = ["ssh", KLIPPER_HOST, KLIPPY_COMMAND]
    print(" ".join(command))
    # Keep session detached from chat-side pipes to avoid backpressure stalls.
    return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)


def run_gcode_via_moonraker(gcode: list[str], duration_seconds: int, client) -> None:
    client.ensure_ready()

    command_timeout = max(20.0, float(duration_seconds) + 15.0)
    for line in gcode:
        client.run_gcode_script(line, timeout_seconds=command_timeout)
        state = client.get_state()
        if state not in {"ready", "printing"}:
            error_context = extract_error_context_from_client(client)
            raise KlippyFailureError(
                f"Moonraker entered unexpected state '{state}' after running gcode: {line}",
                error_context=error_context,
            )

    deadline = time.monotonic() + max(0, duration_seconds)
    while time.monotonic() < deadline:
        state = client.get_state()
        if state not in {"ready", "printing"}:
            error_context = extract_error_context_from_client(client)
            raise KlippyFailureError(
                f"Moonraker entered unexpected state '{state}' while waiting",
                error_context=error_context,
            )
        time.sleep(0.5)


def terminate_process(process: subprocess.Popen[str], timeout_seconds: float = 5.0) -> None:
    if process.poll() is None:
        process.kill()
    try:
        process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        # Local ssh wrapper may linger briefly after remote kill; ignore after best effort.
        pass


def extract_error_context_from_client(client) -> str:
    log_content = client.get_klippy_log()
    return extract_error_context_from_log(log_content)

def extract_error_context_from_local_log() -> str:
    try:
        log_content = local_klippy_log_path().read_text(encoding="utf-8")
        return extract_error_context_from_log(log_content)
    except Exception as e:
        print(f"Failed to read local klippy.log for error context: {e}")
        return ""

def extract_error_context_from_log(log_content: str) -> str:
    lines = log_content.splitlines()
    if not lines:
        return ""

    # Find the most recent startup marker
    startup_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if "Starting Klippy" in lines[i] or "HostEndpoint start_args" in lines[i] or "klippy startup" in lines[i].lower():
            startup_idx = i
            break

    # Extract lines from startup (or beginning) to end
    start = startup_idx + 1 if startup_idx >= 0 else 0
    log_lines = lines[start:]

    # Extract error-relevant lines (strict patterns to avoid false positives)
    error_patterns = [
        r"\[ERROR\]",      # Explicit log level
        r"\[CRITICAL\]",
        r"^(.*)Error(:|s)?",  # Error at start or Error: / Errors
        r"Traceback",
        r"Exception",
        r"Timer too close",  # Known Klipper issue
    ]
    error_lines = []
    for line in log_lines:
        for pattern in error_patterns:
            if pattern.lower() in line.lower() or "exception" in line.lower() or "traceback" in line.lower():
                error_lines.append(line)
                break

    if error_lines:
        return "\n--- klippy.log error context ---\n" + "\n".join(error_lines)
    return ""

def run_scenario(gcode_lines: list[str], duration_seconds: int) -> Path:
    from moonraker_client import MoonrakerClient

    ensure_no_running_klippy()
    remove_remote_klippy_log()

    client = None
    process = start_remote_klippy_session()
    try:
        client = MoonrakerClient(MOONRAKER_URL)
        run_gcode_via_moonraker(gcode_lines, duration_seconds, client)
    except KlippyFailureError:
        # Already has error context attached, just re-raise
        raise
    except RuntimeError as e:
        # Other RuntimeErrors: attempt to attach error context
        error_context = extract_error_context_from_client(client)
        if error_context:
            raise KlippyFailureError(str(e), error_context=error_context) from e
        raise
    finally:
        ensure_no_running_klippy()
        terminate_process(process)
        if client is not None:
            local_klippy_log_path().write_text(client.get_klippy_log(), encoding="utf-8")

    return local_klippy_log_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invoke run-test-rig scenario on remote klippy host")
    parser.add_argument("-DurationSeconds", "--duration-seconds", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gcode = read_gcode_lines()
    try:
        run_scenario(gcode, args.duration_seconds)
    except KlippyFailureError:
        # Already has error context, just propagate
        raise
    except RuntimeError as e:
        # Try to add error context if not already present
        error_context = extract_error_context_from_local_log()
        if error_context:
            raise KlippyFailureError(str(e), error_context=error_context) from e
        raise
    print("Scenario run completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
