---
name: run-test-rig
description: "Use for real hardware validation on Happy-Hare test rig. Export code, run invoke_test_rig.py, inspect `klippy.log`, verify BLDC/sync behavior. Triggers: test on test rig, real hardware validation, run on test rig, MMU_LOAD failure reproduction, klippy.log verification."
argument-hint: "Describe scenario and evidence target (example: MMU_LOAD homing, BLDC gear+extruder concurrency, timeout regression, expected/forbidden log line)."
user-invocable: true
---

# Run Test Rig (Real Hardware)

## When To Use
- Need real-hardware validation beyond static checks.
- Change compiles local, behavior still needs remote klippy proof.
- Need reproduce or clear runtime failure signature from `klippy.log`.
- Need evidence for BLDC/sync path behavior in MMU moves.

## Outcome
Run narrowest real-hardware test using `invoke_test_rig.py` (or compatibility wrapper `Invoke-TestRig.ps1`). Collect pass/fail evidence from script output + `klippy.log`. Report acceptance verdict.

## Scripts
- **Export code (primary)**: `.github/skills/run-test-rig/export_to_test_rig.py` — copies Happy-Hare extras and printer_data config to the rig. Run before first test of a session or after code changes.
- **Run test (primary)**: `.github/skills/run-test-rig/invoke_test_rig.py` — reads GCode from `.github/skills/run-test-rig/startup.gcode`, injects it into rig `printer.cfg`, auto-prepends `SET_KINEMATIC_POSITION X=150 Y=150 Z=25 SET_HOMED=XYZ`, `MMU_TEST_CONFIG LOG_LEVEL=4`, and `MMU_SELECT GATE=0` before the startup GCode; starts klippy, waits `-DurationSeconds`, kills it, downloads `klippy.log` to `.github/skills/run-test-rig/klippy.log`.
- **Download log only (primary)**: `.github/skills/run-test-rig/download_klippy_log.py` — downloads current klippy log from Moonraker and overwrites local `.github/skills/run-test-rig/klippy.log` without running a scenario.
- **Compatibility wrappers**: `.github/skills/run-test-rig/Export-ToTestRig.ps1` and `.github/skills/run-test-rig/Invoke-TestRig.ps1` call Python scripts above.

## Procedure
1. Confirm scope and expected evidence.
- Capture exact scenario and acceptance target.
- Define evidence: command success, expected log pattern, forbidden log pattern, no traceback.

2. Validate and export code to rig.
- If any Python file was edited, syntax-check changed runtime files first (run from `Happy-Hare` root):
  `$changed = @(git diff --name-only; git diff --name-only --cached; git ls-files --others --exclude-standard) | Sort-Object -Unique`
  `$py = $changed | Where-Object { ($_ -like 'extras/*.py' -or $_ -like 'extras/**/*.py' -or $_ -like 'components/*.py' -or $_ -like 'components/**/*.py') -and (Test-Path $_) }`
  `$py | ForEach-Object { python -m py_compile $_ }`
  Fix any errors before proceeding. Do not export with syntax errors.
- Run (primary): `python .github/skills/run-test-rig/export_to_test_rig.py`
- Run (compat): `. .github/skills/run-test-rig/Export-ToTestRig.ps1`
- Skip export (and py_compile) if code unchanged since last export this session.

3. Build the GCode for the narrow test.
- Include only commands needed to reproduce/validate behavior.
- Write GCode to `.github/skills/run-test-rig/startup.gcode` (one command per line).
- `invoke_test_rig.py` automatically prepends `SET_KINEMATIC_POSITION X=150 Y=150 Z=25 SET_HOMED=XYZ`, `MMU_TEST_CONFIG LOG_LEVEL=4`, and `MMU_SELECT GATE=0` — do NOT include any of those in `startup.gcode`.
- If manual setup needed before run (insert filament, clear jam, reset hardware), stop and ask user to do setup first.

4. Run remote klippy and collect output.
- Run (primary): `python .github/skills/run-test-rig/invoke_test_rig.py -DurationSeconds <N>`
- Run (compat): `. .github/skills/run-test-rig/Invoke-TestRig.ps1 -DurationSeconds <N>`
- Choose `-DurationSeconds` = estimated scenario runtime × 1.5 (include all dwells, move durations, and startup time). Round up to nearest second.
- Script removes previous remote `klippy.log`, runs klippy, downloads fresh log locally.
- Inspect `.github/skills/run-test-rig/klippy.log`.
- Validate evidence with explicit checks (PowerShell examples):
  `Select-String -Path .github/skills/run-test-rig/klippy.log -SimpleMatch '<expected pattern>'`
  `Select-String -Path .github/skills/run-test-rig/klippy.log -SimpleMatch 'Unknown command:','Traceback (most recent call last)'`
- Enforce no-command-error rule: fail immediately on `Unknown command:` or traceback.

5. Evaluate with decision points.
- If command unknown: fail run. Fix GCode or code before rerun.
- If startup/init not complete before test command: increase `-DurationSeconds` or add dwell.
- If expected pattern missing: mark inconclusive unless equivalent evidence exists.
- If forbidden pattern appears (or traceback): fail and report first causal lines.

6. Report with acceptance verdict.
- State pass/fail/inconclusive.
- Provide key evidence lines, not full raw log.
- Include exact next action if inconclusive.

## Decision Points
- Re-run criteria:
  - Re-run when logs show command mismatch, startup ordering issue, or stale log contamination.
  - Do not re-run unchanged scenario repeatedly after root cause identified.

- No-error policy:
  - Any run that emits `Unknown command:` or traceback is a failed validation, not a soft warning.

- Evidence threshold:
  - Pass requires positive expected-behavior evidence + no fatal errors.
  - Inconclusive if environment/macro mismatch blocks target path.

## Completion Checklist
- [ ] If Python files were edited, `python -m py_compile` passed on all changed files before export.
- [ ] Code exported to rig via `export_to_test_rig.py` (or wrapper `Export-ToTestRig.ps1`).
- [ ] `invoke_test_rig.py` used with `-DurationSeconds` = estimated runtime × 1.5.
- [ ] Expected and forbidden log patterns were verified with explicit log search commands.
- [ ] If non-blocking gcodes tested, `startup.gcode` includes `G4 P<ms>` dwell to allow completion.
- [ ] If manual hardware setup required, user was asked to perform setup before execution.
- [ ] No `Unknown command:` lines were emitted.
- [ ] No traceback lines were emitted.
- [ ] Expected evidence present for target scenario.
- [ ] Forbidden signature/traceback absent.
- [ ] Verdict reported: pass/fail/inconclusive.
- [ ] Verdict not inconclusive, or next rerun action documented.

## Reference Command Pattern
- Export code (primary): `python .github/skills/run-test-rig/export_to_test_rig.py`
- Export code (compat): `. .github/skills/run-test-rig/Export-ToTestRig.ps1`
- Write test GCode to `.github/skills/run-test-rig/startup.gcode` (one GCode command per line)
- Run scenario (primary): `python .github/skills/run-test-rig/invoke_test_rig.py -DurationSeconds <N>`
- Run scenario (compat): `. .github/skills/run-test-rig/Invoke-TestRig.ps1 -DurationSeconds <N>`
- Download log only (primary): `python .github/skills/run-test-rig/download_klippy_log.py`
- Verify evidence: `Select-String -Path .github/skills/run-test-rig/klippy.log -SimpleMatch '<expected pattern>'`
- Verify no fatal errors: `Select-String -Path .github/skills/run-test-rig/klippy.log -SimpleMatch 'Unknown command:','Traceback (most recent call last)'`
- Log location: `.github/skills/run-test-rig/klippy.log` (downloaded automatically by `invoke_test_rig.py`)
- Forbidden: startup.gcode, VS Code tasks, ad-hoc SSH commands for test execution.
