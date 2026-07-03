from __future__ import annotations

import subprocess
import sys
from pathlib import Path

KLIPPER_URL = "klippy-test"


def script_root() -> Path:
    return Path(__file__).resolve().parent


def happy_hare_root() -> Path:
    return script_root().parents[2]


def printer_data_root() -> Path:
    return happy_hare_root().parent / "printer_data"


def run_ssh(command: str, check: bool = True) -> None:
    full_command = ["ssh", KLIPPER_URL, command]
    print(" ".join(full_command))
    result = subprocess.run(full_command, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"SSH command failed: {command}\n{result.stderr}")


def convert_to_local_path(remote_path: str) -> Path:
    if remote_path.startswith("Happy-Hare"):
        return Path(remote_path.replace("Happy-Hare", str(happy_hare_root()))).resolve()

    if remote_path.startswith("printer_data/config/mmu"):
        return Path(remote_path.replace("printer_data/config/mmu", str(printer_data_root()))).resolve()

    raise ValueError(f"Unknown remote pattern: {remote_path}")


def source_files(path: Path) -> list[str]:
    return [str(entry) for entry in path.glob("*.*") if entry.is_file()]


def copy_to_ssh(source: Path, destination: str) -> None:
    run_ssh(f"rm {destination}/*.*", check=False)

    files = source_files(source)
    if not files:
        return

    command = ["scp", *files, f"{KLIPPER_URL}:{destination}/"]
    print(" ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to copy {source} to {KLIPPER_URL}:{destination}\n{result.stderr}")


def main() -> int:
    run_ssh("pgrep -f klippy-env/bin/python && kill `pgrep -f klippy-env/bin/python`", check=False)

    remote_paths = [
        "printer_data/config/mmu",
        "printer_data/config/mmu/base",
        "printer_data/config/mmu/addons",
        "printer_data/config/mmu/optional",
        "Happy-Hare/extras",
        "Happy-Hare/extras/mmu",
    ]

    for remote_path in remote_paths:
        local_path = convert_to_local_path(remote_path)
        copy_to_ssh(local_path, remote_path)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
