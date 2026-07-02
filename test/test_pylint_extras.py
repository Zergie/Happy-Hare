from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


@dataclass(frozen=True)
class Scenario:
    label: str
    file_path: Path
    is_new_file: bool
    changed_lines: set[int]


HUNK_HEADER_PATTERN = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
PYLINT_TEXT_PATTERN = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):\d+:\s+(?P<code>[A-Z]\d+):\s+(?P<message>.+?)\s+\((?P<symbol>[^)]+)\)\s*$"
)

# Optional suppressions keyed by pylint symbol, then relative path and full source line (stripped).
DIAGNOSTIC_LINE_IGNORES: dict[str, dict[str, list[str]]] = {
    "too-many-positional-arguments": {
        "extras/mmu/mmu.py": [
            "def trace_bldc_filament_move(self, trace_str, dist, speed, accel, motor, homing_move, endstop_name):",
        ],
        "extras/mmu_machine.py": [
            "def add_extra_endstop(self, pin, name, register=True, bind_rail_steppers=True, mcu_endstop=None):",
        ],
        "extras/mmu/mmu_gear_bldc.py": [
            "def __init__(self, print_time, speed_mm_s, pid_enable, direction=1, duration=0.):",
        ],
    },
    "import-outside-toplevel": {
        "extras/mmu/mmu_gear_bldc.py": [
            "from .. import output_pin",
            "from ..mmu_espooler import GCodeRequestQueue as FallbackGCodeRequestQueue",
        ]
    },
    "protected-access": {
        "extras/mmu/mmu_gear_bldc.py": [
            "extruder._hh_bldc_original_process_move = process_move",
            "extruder._hh_bldc_process_move_owner = self",
        ],
    },
    "no-value-for-parameter": {
        "extras/mmu/mmu_gear_bldc.py": [
            "self.mcu_pwm_pin = MCU_queued_pwm(pwm_pin_params)",
        ],
    }
}

_FILE_LINE_CACHE: dict[str, list[str]] = {}
_IGNORE_USAGE_STATE: dict[str, dict[str, int | bool]] = {}
_IGNORE_MATCHED_LINES: dict[tuple[str, str], set[str]] = {}


def _mark_ignore_file_read(scenario_label: str) -> None:
    state = _IGNORE_USAGE_STATE.setdefault(scenario_label, {"file_read": False, "used": 0})
    state["file_read"] = True


def _mark_ignore_used(scenario_label: str) -> None:
    state = _IGNORE_USAGE_STATE.setdefault(scenario_label, {"file_read": False, "used": 0})
    state["used"] = int(state["used"]) + 1


def _mark_ignore_line_used(scenario_label: str, symbol: str, source_line: str) -> None:
    key = (scenario_label, symbol)
    matched = _IGNORE_MATCHED_LINES.setdefault(key, set())
    matched.add(source_line)


def _raise_if_stale_or_unused_ignore_lines(
    scenario: Scenario, diagnostics: list[tuple[int, str, str, str]]
) -> None:
    state = _IGNORE_USAGE_STATE.get(scenario.label)
    if not state or not bool(state.get("file_read")):
        return

    symbols_present = {symbol for _line, symbol, _message, _code_or_type in diagnostics}
    source_lines = {
        line.strip() for line in scenario.file_path.read_text(encoding="utf-8").splitlines() if line.strip()
    }

    issues: list[str] = []
    for symbol, symbol_rules in DIAGNOSTIC_LINE_IGNORES.items():
        ignored_lines = symbol_rules.get(scenario.label)
        if not ignored_lines or "*" in ignored_lines:
            continue

        matched_lines = _IGNORE_MATCHED_LINES.get((scenario.label, symbol), set())
        for ignored_line in ignored_lines:
            if ignored_line not in source_lines:
                issues.append(
                    "symbol=%s line not found in file: %s" % (symbol, ignored_line)
                )
                continue
            if symbol not in symbols_present:
                issues.append(
                    "symbol=%s has no diagnostics, ignore line unused: %s" % (symbol, ignored_line)
                )
                continue
            if ignored_line not in matched_lines:
                issues.append(
                    "symbol=%s diagnostics present, but ignore line not used: %s" % (symbol, ignored_line)
                )

    if issues:
        raise AssertionError(
            "DIAGNOSTIC_LINE_IGNORES has stale/unused entries for %s:\n%s"
            % (scenario.label, "\n".join(issues))
        )


def _raise_if_ignore_rules_not_used(scenario: Scenario) -> None:
    state = _IGNORE_USAGE_STATE.get(scenario.label)
    if not state:
        return
    if bool(state.get("file_read")) and int(state.get("used", 0)) == 0:
        raise AssertionError(
            "DIAGNOSTIC_LINE_IGNORES contains entries for %s, but none were used after reading source lines. "
            "Remove stale ignore entries or update line strings."
            % scenario.label
        )


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _get_pylintrc_path() -> Path:
    return _get_repo_root() / "extras" / ".pylintrc"


def _get_workspace_root() -> Path:
    return _get_repo_root().parents[2]


def _get_lint_target_root() -> Path:
    return _get_workspace_root() / "klipper" / "klippy"


def _remove_path(path: Path) -> None:
    is_junction = getattr(path, "is_junction", lambda: False)()
    if path.is_symlink() or is_junction or path.is_file():
        path.unlink()
        return
    if path.exists():
        shutil.rmtree(path)


def _link_directory(target: Path, source: Path) -> None:
    if sys.platform == "win32":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target), str(source)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            output = "\n".join(
                line for line in [result.stdout.strip(), result.stderr.strip()] if line
            )
            raise AssertionError("Failed to create junction %s -> %s\n%s" % (target, source, output))
        return
    target.symlink_to(source, target_is_directory=True)


def _read_link_target(path: Path) -> str | None:
    if not path.exists() and not path.is_symlink():
        return None

    if sys.platform == "win32":
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-Item -LiteralPath '%s' -Force).Target" % str(path).replace("'", "''"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        target = result.stdout.strip()
        return target or None

    return str(path.resolve(strict=False))


def _matches_link_target(path: Path, source: Path) -> bool:
    target = _read_link_target(path)
    if target is None:
        return False
    return Path(target).resolve() == source.resolve()


def _ensure_lint_target_link(scenario: Scenario) -> Path:
    klippy_root = _get_lint_target_root()
    if not klippy_root.exists():
        raise AssertionError("Missing Klipper klippy dir: %s" % klippy_root)

    target_extras = klippy_root / "extras"
    if not target_extras.exists():
        raise AssertionError("Missing extras dir: %s" % target_extras)

    relative_path = Path(scenario.label)
    source_path = scenario.file_path

    if len(relative_path.parts) > 1 and relative_path.parts[1] == "mmu":
        source_dir = source_path.parent
        target_dir = target_extras / "mmu"
        if not _matches_link_target(target_dir, source_dir):
            if target_dir.exists() or target_dir.is_symlink():
                _remove_path(target_dir)
            _link_directory(target_dir, source_dir)
        return klippy_root

    target_file = target_extras / relative_path.name
    if not _matches_link_target(target_file, source_path):
        if target_file.exists() or target_file.is_symlink():
            _remove_path(target_file)
        target_file.symlink_to(source_path)
    return klippy_root


def _scenario_module_name(scenario: Scenario) -> str:
    relative_path = Path(scenario.label)
    module_path = relative_path.with_suffix("")
    return ".".join(module_path.parts)


def _run_git_command(args: list[str]) -> str:
    result = subprocess.run(
        ["git"] + args,
        cwd=str(_get_repo_root()),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout

    output = "\n".join(line for line in [result.stdout.strip(), result.stderr.strip()] if line)
    raise AssertionError(
        "git command failed: git %s\n%s\n"
        "This test uses local remote/main and does not update refs (no fetch/pull)."
        % (" ".join(args), output)
    )


def _get_diff_base() -> str:
    return _run_git_command(["merge-base", "remote/main", "HEAD"]).strip()


def _get_changed_python_files() -> dict[str, bool]:
    diff_base = _get_diff_base()
    output = _run_git_command(["diff", "--name-status", "--diff-filter=AM", diff_base, "--", "extras"])
    changed_files: dict[str, bool] = {}

    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0].strip()
        file_path = parts[1].strip().replace("\\", "/")
        if not file_path.endswith(".py"):
            continue
        changed_files[file_path] = status == "A"

    return changed_files


def _get_changed_lines_by_file() -> dict[str, set[int]]:
    diff_base = _get_diff_base()
    output = _run_git_command(["diff", "-U0", "--diff-filter=AM", diff_base, "--", "extras"])
    changed_lines: dict[str, set[int]] = {}
    current_file = None
    in_hunk = False
    current_new_line = None

    for line in output.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip().replace("\\", "/")
            changed_lines.setdefault(current_file, set())
            in_hunk = False
            current_new_line = None
            continue
        if line.startswith("+++"):
            current_file = None
            in_hunk = False
            current_new_line = None
            continue

        match = HUNK_HEADER_PATTERN.match(line)
        if match is not None and current_file is not None:
            in_hunk = True
            current_new_line = int(match.group(1))
            continue

        if not in_hunk or current_file is None or current_new_line is None:
            continue

        if line.startswith("+") and not line.startswith("+++"):
            changed_lines[current_file].add(current_new_line)
            current_new_line += 1
            continue
        if line.startswith("-") and not line.startswith("---"):
            continue

        # Context line in a non-U0 diff.
        current_new_line += 1

    return changed_lines


def _get_ignored_filenames(pylintrc_path: Path) -> set[str]:
    parser = ConfigParser()
    parser.read(pylintrc_path, encoding="utf-8")
    raw = parser.get("MASTER", "ignore", fallback="")
    return {part.strip() for part in raw.split(",") if part.strip()}


def _parse_pylint_diagnostics(stdout_text: str, fallback_text: str) -> list[tuple[int, str, str, str]]:
    diagnostics: list[tuple[int, str, str, str]] = []

    try:
        payload = json.loads(stdout_text)
    except Exception:
        payload = None

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            line = item.get("line")
            if not isinstance(line, int):
                continue
            diagnostics.append(
                (
                    line,
                    str(item.get("symbol", "")),
                    str(item.get("message", "")),
                    str(item.get("type", "")),
                )
            )
        if diagnostics:
            return diagnostics

    for line in fallback_text.splitlines():
        match = PYLINT_TEXT_PATTERN.match(line.strip())
        if match is None:
            continue
        diagnostics.append(
            (
                int(match.group("line")),
                match.group("symbol"),
                match.group("message"),
                match.group("code"),
            )
        )

    return diagnostics


def _format_diagnostics(diagnostics: list[tuple[int, str, str, str]]) -> str:
    lines = []
    for line_number, symbol, message, code_or_type in diagnostics:
        lines.append("L%d (%s) %s: %s" % (line_number, code_or_type, symbol, message))
    return "\n".join(lines)


def _get_source_line(file_path: Path, line_number: int) -> str:
    cache_key = str(file_path)
    lines = _FILE_LINE_CACHE.get(cache_key)
    if lines is None:
        lines = file_path.read_text(encoding="utf-8").splitlines()
        _FILE_LINE_CACHE[cache_key] = lines

    if line_number < 1 or line_number > len(lines):
        return ""
    return lines[line_number - 1].strip()


def _is_ignored_diagnostic(scenario: Scenario, diagnostic: tuple[int, str, str, str]) -> bool:
    line_number, symbol, _message, _code_or_type = diagnostic
    symbol_rules = DIAGNOSTIC_LINE_IGNORES.get(symbol)
    if not symbol_rules:
        return False

    ignored_lines = symbol_rules.get(scenario.label)
    if not ignored_lines:
        return False
    if "*" in ignored_lines:
        _mark_ignore_used(scenario.label)
        return True
    _mark_ignore_file_read(scenario.label)
    source_line = _get_source_line(scenario.file_path, line_number)
    if source_line in ignored_lines:
        _mark_ignore_used(scenario.label)
        _mark_ignore_line_used(scenario.label, symbol, source_line)
        return True
    return False


def _build_scenarios() -> list[Scenario]:
    repo_root = _get_repo_root()
    ignored_filenames = _get_ignored_filenames(_get_pylintrc_path())
    changed_files = _get_changed_python_files()
    changed_lines_by_file = _get_changed_lines_by_file()
    scenarios: list[Scenario] = []

    for relative_path, is_new_file in sorted(changed_files.items()):
        file_path = repo_root / relative_path
        if not file_path.exists():
            continue
        if file_path.name in ignored_filenames:
            continue
        scenarios.append(
            Scenario(
                label=relative_path,
                file_path=file_path,
                is_new_file=is_new_file,
                changed_lines=set() if is_new_file else changed_lines_by_file.get(relative_path, set()),
            )
        )

    return scenarios


def _build_failure_output(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(line for line in [result.stdout.strip(), result.stderr.strip()] if line)


def _parse_pylint_diagnostics_with_module(
    stdout_text: str,
) -> dict[str, list[tuple[int, str, str, str]]]:
    """Parse Pylint JSON output and group diagnostics by module name."""
    diagnostics_by_module: dict[str, list[tuple[int, str, str, str]]] = {}

    try:
        payload = json.loads(stdout_text)
    except Exception:
        return diagnostics_by_module

    if not isinstance(payload, list):
        return diagnostics_by_module

    for item in payload:
        if not isinstance(item, dict):
            continue
        module = item.get("module", "")
        line = item.get("line")
        if not isinstance(line, int):
            continue

        if module not in diagnostics_by_module:
            diagnostics_by_module[module] = []
        diagnostics_by_module[module].append(
            (
                line,
                str(item.get("symbol", "")),
                str(item.get("message", "")),
                str(item.get("type", "")),
            )
        )

    return diagnostics_by_module


PYLINTRC_PATH = _get_pylintrc_path()
SCENARIOS = _build_scenarios()

# Cache for single-run Pylint results, populated on first test execution
_PYLINT_RESULTS_CACHE: dict[str, list[tuple[int, str, str, str]]] | None = None


def _ensure_pylint_results_cached() -> dict[str, list[tuple[int, str, str, str]]]:
    """Run Pylint once for all scenarios and cache results by module name."""
    global _PYLINT_RESULTS_CACHE

    if _PYLINT_RESULTS_CACHE is not None:
        return _PYLINT_RESULTS_CACHE

    _PYLINT_RESULTS_CACHE = {}

    if not SCENARIOS:
        return _PYLINT_RESULTS_CACHE

    if not PYLINTRC_PATH.exists():
        raise AssertionError("Missing pylint rcfile: %s" % PYLINTRC_PATH)

    # Set up all symlinks and collect module names
    klippy_root = _get_lint_target_root()
    lint_targets = []

    for scenario in SCENARIOS:
        _ensure_lint_target_link(scenario)
        lint_target = _scenario_module_name(scenario)
        lint_targets.append(lint_target)

    # Run pylint once with all targets
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pylint",
            "--output-format=json",
            "--score=n",
            "--rcfile",
            str(PYLINTRC_PATH),
            *lint_targets,
        ],
        cwd=str(klippy_root),
        capture_output=True,
        text=True,
        check=False,
    )

    output = _build_failure_output(result)
    if "No module named pylint" in output:
        raise AssertionError("pylint is required but not installed\n%s" % output)

    # Parse all diagnostics grouped by module
    _PYLINT_RESULTS_CACHE = _parse_pylint_diagnostics_with_module(result.stdout)
    if not _PYLINT_RESULTS_CACHE and result.returncode != 0:
        raise AssertionError("pylint failed and diagnostics could not be parsed\n%s" % output)

    return _PYLINT_RESULTS_CACHE


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[scenario.label for scenario in SCENARIOS])
@pytest.mark.timeout(60)
def test_pylint_each_extras_file(scenario: Scenario) -> None:
    """Validate each changed file against cached Pylint results."""
    _IGNORE_USAGE_STATE.pop(scenario.label, None)
    for key in list(_IGNORE_MATCHED_LINES.keys()):
        if key[0] == scenario.label:
            del _IGNORE_MATCHED_LINES[key]

    # Ensure Pylint has run and cache is populated
    diagnostics_by_module = _ensure_pylint_results_cached()

    lint_target = _scenario_module_name(scenario)
    raw_diagnostics = diagnostics_by_module.get(lint_target, [])

    diagnostics = [
        diagnostic for diagnostic in raw_diagnostics if not _is_ignored_diagnostic(scenario, diagnostic)
    ]
    _raise_if_stale_or_unused_ignore_lines(scenario, raw_diagnostics)
    _raise_if_ignore_rules_not_used(scenario)

    if not diagnostics:
        return

    if scenario.is_new_file:
        raise AssertionError(
            "pylint failed for %s on changed lines since remote/main (%d/%d diagnostics)\n%s"
            % (
                scenario.label,
                len(diagnostics),
                len(diagnostics),
                _format_diagnostics(diagnostics),
            )
        )

    changed_line_diagnostics = [
        diagnostic for diagnostic in diagnostics if diagnostic[0] in scenario.changed_lines
    ]
    if not changed_line_diagnostics:
        return

    raise AssertionError(
        "pylint failed for %s on changed lines since remote/main (%d/%d diagnostics)\n%s"
        % (
            scenario.label,
            len(changed_line_diagnostics),
            len(diagnostics),
            _format_diagnostics(changed_line_diagnostics),
        )
    )

