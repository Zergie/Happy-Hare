from __future__ import annotations

from pathlib import Path
import subprocess
import sys


_has_exported_run_test_rig_code = False


def _is_tools_helper_path(path_text: str) -> bool:
    return "/tools_helper/" in Path(path_text).as_posix()


def _export_run_test_rig_code() -> None:
    export_script = Path(__file__).resolve().parent.parent / "export_to_test_rig.py"
    result = subprocess.run(
        [sys.executable, str(export_script)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        output = (result.stdout + "\n" + result.stderr).strip()
        raise RuntimeError("run-test-rig export failed before test execution. Output:\n%s" % output)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "xdist_group(name): run tests in the same xdist worker group",
    )


def pytest_collection_modifyitems(config, items):
    # Only relevant when running with pytest-xdist; harmless otherwise.
    for item in items:
        item_path = Path(str(item.fspath)).as_posix()
        if _is_tools_helper_path(item_path):
            continue
        item.add_marker("xdist_group", "sequential")


def pytest_runtest_setup(item):
    global _has_exported_run_test_rig_code
    item_path = str(item.fspath)
    if _is_tools_helper_path(item_path):
        return
    if _has_exported_run_test_rig_code:
        return
    _export_run_test_rig_code()
    _has_exported_run_test_rig_code = True
