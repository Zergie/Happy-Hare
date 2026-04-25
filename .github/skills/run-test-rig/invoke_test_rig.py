from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

KLIPPER_URL = "klippy-test"
KLIPPY_LOG_REMOTE = "/home/user/printer_data/logs/klippy.log"
KLIPPY_COMMAND = (
    "/home/user/klippy-env/bin/python /home/user/klipper/klippy/klippy.py "
    "/home/user/printer_data/config/printer.cfg "
    "-I /home/user/printer_data/comms/klippy.serial "
    "-l /home/user/printer_data/logs/klippy.log "
    "-a /home/user/printer_data/comms/klippy.sock;"
)


def script_root() -> Path:
    return Path(__file__).resolve().parent


def startup_gcode_path() -> Path:
    return script_root() / "startup.gcode"


def local_klippy_log_path() -> Path:
    return script_root() / "klippy.log"


def run_ssh(command: str, check: bool = True) -> str:
    full_command = ["ssh", KLIPPER_URL, command]
    print(" ".join(full_command))
    result = subprocess.run(full_command, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"SSH command failed: {command}\n{result.stderr}")
    return result.stdout


def run_scp(source: str, destination: str) -> None:
    command = ["scp", source, destination]
    print(" ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {' '.join(command)}\n{result.stderr}")


def read_startup_gcode_lines() -> list[str]:
    content = startup_gcode_path().read_text(encoding="utf-8")
    return content.splitlines()


def build_printer_cfg_with_startup_block(printer_cfg_text: str, startup_lines: list[str]) -> str:
    lines = printer_cfg_text.splitlines()
    output: list[str] = []
    skipping_old_startup = False

    for line in lines:
        if skipping_old_startup and line.startswith("#*#"):
            skipping_old_startup = False
            output.append(line)
            continue

        if line.startswith("[delayed_gcode startup]"):
            skipping_old_startup = True
            output.append(line)
            output.append("initial_duration: 1")
            output.append("gcode:")
            output.append("  SET_KINEMATIC_POSITION X=150 Y=150 Z=25 SET_HOMED=XYZ")
            output.append("  MMU_TEST_CONFIG LOG_LEVEL=4")
            output.append("  MMU_SELECT GATE=0")
            output.extend([f"  {entry}" for entry in startup_lines])
            output.append("")
            continue

        if not skipping_old_startup:
            output.append(line)

    return "\n".join(output) + "\n"


def write_temp_printer_cfg(content: str) -> Path:
    temp_path = Path(tempfile.gettempdir()) / "printer.cfg"
    temp_path.write_text(content, encoding="utf-8")
    return temp_path


def ensure_no_running_klippy() -> None:
    run_ssh("pgrep -f klippy-env/bin/python && kill `pgrep -f klippy-env/bin/python`", check=False)


def remove_remote_klippy_log() -> None:
    run_ssh(f"rm {KLIPPY_LOG_REMOTE}", check=False)


def run_remote_klippy(duration_seconds: int) -> tuple[int, str, str]:
    command = ["ssh", KLIPPER_URL, KLIPPY_COMMAND]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        stdout, stderr = process.communicate(timeout=duration_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        ensure_no_running_klippy()

    return process.returncode, stdout, stderr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invoke run-test-rig scenario on remote klippy host")
    parser.add_argument("-DurationSeconds", "--duration-seconds", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    startup_lines = read_startup_gcode_lines()
    printer_cfg_text = run_ssh("cat printer_data/config/printer.cfg")
    updated_cfg = build_printer_cfg_with_startup_block(printer_cfg_text, startup_lines)
    temp_cfg_path = write_temp_printer_cfg(updated_cfg)

    run_scp(str(temp_cfg_path), f"{KLIPPER_URL}:printer_data/config/printer.cfg")
    ensure_no_running_klippy()
    remove_remote_klippy_log()

    exit_code, stdout, stderr = run_remote_klippy(args.duration_seconds)
    run_scp(f"{KLIPPER_URL}:{KLIPPY_LOG_REMOTE}", str(local_klippy_log_path()))

    print(f"Test rig process exited with code {exit_code}. Output:")
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
