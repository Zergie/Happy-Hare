from __future__ import annotations

import sys
import traceback
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

MOONRAKER_URL = "http://klippy-test:7125"


def script_root() -> Path:
    return SCRIPT_ROOT


def local_klippy_log_path() -> Path:
    return script_root() / "klippy.log"


def download_klippy_log() -> Path:
    from moonraker_client import MoonrakerClient

    client = MoonrakerClient(MOONRAKER_URL)
    log_content = client.get_klippy_log()

    output_path = local_klippy_log_path()
    output_path.write_text(log_content, encoding="utf-8")
    return output_path


def main() -> int:
    output_path = download_klippy_log()
    print(str(output_path))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)
