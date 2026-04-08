---
name: run-test-rig
description: "Use when testing Happy-Hare changes on the real test rig hardware (export, run klippy remotely, execute MMU startup gcode, and validate klippy.log outcomes including BLDC/sync behavior). Trigger phrases: test on test rig, real hardware validation, run on test rig, MMU_LOAD failure reproduction, klippy.log verification."
argument-hint: "Describe the scenario to validate (for example: MMU_LOAD toolhead homing, BLDC gear+extruder concurrency, timeout regression, or a specific expected/forbidden log line)."
user-invocable: true
---

# Run Test Rig (Real Hardware)

## When To Use
- You need real-hardware validation beyond static checks.
- A change compiles locally but behavior must be verified against remote klippy.
- You need to reproduce or clear a runtime failure signature from `klippy.log`.
- You need evidence for BLDC/sync path behavior in MMU moves.

## Outcome
Run the narrowest real-hardware test using VS Code tasks only, collect actionable pass/fail evidence from task output and `klippy.log`, and report whether acceptance criteria are met.

## Procedure
1. Confirm scope and expected evidence.
- Capture exact scenario and acceptance target.
- Define required evidence: specific command success, expected log pattern, forbidden log pattern, and no traceback.

2. Export code to rig first.
- Use VS Code task in `klipper` workspace: `Export Code to Test Rig`.
- Confirm remote files were copied successfully before running tests.

3. Choose execution path (tasks only).
- Preferred quick path: task `Run Test on Test Rig` when default startup sequence matches the scenario.
- Custom validation path: edit `startup.gcode` in this skill folder, export it with task `Export Startup.gcode to Test Rig`, then run task `Run Startup.gcode on Test Rig` — this task deletes the previous `klippy.log` and starts a fresh one before running klippy.
- Do not run ad-hoc shell commands directly for test execution.

4. Build startup gcode for the narrow test.
- Include only setup + commands needed to reproduce/validate behavior.
- Always put `SET_KINEMATIC_POSITION X=150 Y=150 Z=25 SET_HOMED=XYZ` as the very first line.
- Always follow with `MMU_TEST_CONFIG LOG_LEVEL=4` — this ensures maximum verbosity in the log for every run.
- Always follow that with `MMU_SELECT GATE=0` — all rig tests use gate 0.
- Typical MMU flow: `SET_KINEMATIC_POSITION X=150 Y=150 Z=25 SET_HOMED=XYZ`, `MMU_TEST_CONFIG LOG_LEVEL=4`, `MMU_SELECT GATE=0`, then the targeted move/homing/load command, optional reverse/cleanup move.
- Edit file: `.github/skills/run-test-rig/startup.gcode`.

5. Run remote klippy and collect output.
- Task `Run Startup.gcode on Test Rig` automatically removes the old `klippy.log` before starting, ensuring no stale signal from prior runs.
- Execute klippy once with startup script and capture task exit status.
- Use VS Code task `Read klippy.log on Test Rig` to fetch `klippy.log` after the run.
- Enforce no-command-error rule: fail immediately on `Unknown command:` or traceback.

6. Evaluate with decision points.
- If command is unknown: fail the run and update/extend task command list before rerun.
- If startup/init not complete before test command: treat as sequencing issue, adjust startup gcode order, rerun.
- If expected pattern missing: mark as inconclusive unless equivalent evidence exists.
- If forbidden pattern appears (or traceback appears): fail and report first causal lines.

7. Report with acceptance verdict.
- State pass/fail/inconclusive.
- Provide key evidence lines, not full raw logs.
- Include exact next action if inconclusive.

## Decision Points
- Task path vs custom path:
  - Use `Run Test on Test Rig` for generic sanity checks.
  - Use extended task variants for scenario-specific validation (for example MMU homing mode behavior).

- Re-run criteria:
  - Re-run when logs show command mismatch, startup ordering issues, or stale log contamination.
  - Do not re-run unchanged scenario repeatedly once root cause is identified.

- No-error policy:
  - Test task definitions must include checks for `Unknown command:` and traceback signatures.
  - Any run that emits command errors is a failed validation, not a soft warning.

- Evidence threshold:
  - Pass requires positive evidence for expected behavior plus absence of fatal errors.
  - Inconclusive if environment/macro mismatch prevents exercising target path.

## Completion Checklist
- [ ] Code exported to rig successfully.
- [ ] Scenario script updated in `.github/skills/run-test-rig/startup.gcode`.
- [ ] `Export Startup.gcode to Test Rig` task completed successfully.
- [ ] `Run Startup.gcode on Test Rig` task used to execute scenario (old log deleted automatically).
- [ ] `Read klippy.log on Test Rig` task used to fetch and inspect the fresh log.
- [ ] No `Unknown command:` lines were emitted.
- [ ] No traceback lines were emitted.
- [ ] Expected evidence present for target scenario.
- [ ] Forbidden signature/traceback absent.
- [ ] Verdict reported: pass/fail/inconclusive.
- [ ] Verdict is not inconclusive, or next rerun action is documented.

## Reference Command Pattern
- Export: VS Code task `Export Code to Test Rig`.
- Scenario gcode authoring: edit `.github/skills/run-test-rig/startup.gcode`.
- Scenario gcode deployment: VS Code task `Export Startup.gcode to Test Rig`.
- Run scenario on rig: VS Code task `Run Startup.gcode on Test Rig` — deletes the previous `klippy.log` before starting, so each run produces a fresh log.
- Read log: VS Code task `Read klippy.log on Test Rig` — fetches the current `klippy.log` from the rig.
- Custom run pattern must be implemented as VS Code tasks.
