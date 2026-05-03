from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import pytest


@dataclass(frozen=True)
class Scenario:
    label: str
    file_path: Path


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _build_scenarios() -> list[Scenario]:
    repo_root = _get_repo_root()
    extras_root = repo_root / "extras"
    scenarios: list[Scenario] = []
    for file_path in sorted(extras_root.rglob("*.py")):
        relative_path = file_path.relative_to(repo_root).as_posix()
        scenarios.append(Scenario(label=relative_path, file_path=file_path))
    return scenarios


SCENARIOS = _build_scenarios()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[scenario.label for scenario in SCENARIOS])
def test_py_compile_each_extras_file(scenario: Scenario) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(scenario.file_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return

    output = "\n".join(line for line in [result.stdout.strip(), result.stderr.strip()] if line)
    raise AssertionError(
        "py_compile failed for %s\n%s" % (scenario.label, output)
    )
