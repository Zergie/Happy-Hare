---
name: run-test-rig
description: "Use for real hardware validation on Happy-Hare test rig. Export code, run startup gcode, inspect `klippy.log`, verify BLDC/sync behavior. Triggers: test on test rig, real hardware validation, run on test rig, MMU_LOAD failure reproduction, klippy.log verification."
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
Run narrowest real-hardware test with VS Code tasks only. Collect pass/fail evidence from task output + `klippy.log`. Report acceptance verdict.

## Procedure
1. Confirm scope and expected evidence.
- Capture exact scenario and acceptance target.
- Define evidence: command success, expected log pattern, forbidden log pattern, no traceback.

2. Export code to rig first.
- Use VS Code task in `Happy-Hare` workspace: `Export Code to Test Rig`.
- Confirm remote copy success before test run.

3. Use one execution path (tasks only).
- Use task `Run Startup.gcode on Test Rig` for all test execution now.
- This task executes `.github/skills/run-test-rig/startup.gcode` on the test rig.
- After execution finishes, this task downloads the resulting `klippy.log` to `.github/skills/run-test-rig/klippy.log`.
- Do not run any ad-hoc shell commands or any other SSH commands for test execution.

4. Build startup gcode for the narrow test.
- Include only setup + commands needed to reproduce/validate behavior.
- Always put `SET_KINEMATIC_POSITION X=150 Y=150 Z=25 SET_HOMED=XYZ` as the very first line.
- Always follow with `MMU_TEST_CONFIG LOG_LEVEL=4` — this ensures maximum verbosity in the log for every run.
- Always follow that with `MMU_SELECT GATE=0` — all rig tests use gate 0.
- If test uses non-blocking gcodes, add `G4 P<ms>` dwell in `startup.gcode` so async behavior finishes before klippy exit (example: `G4 P1000` ~= 1s).
- Typical MMU flow: `SET_KINEMATIC_POSITION X=150 Y=150 Z=25 SET_HOMED=XYZ`, `MMU_TEST_CONFIG LOG_LEVEL=4`, `MMU_SELECT GATE=0`, targeted move/homing/load, optional reverse/cleanup move.
- Edit file: `.github/skills/run-test-rig/startup.gcode`.
- If manual setup needed before run (insert filament, clear jam, reset hardware), stop and ask user to do setup first.

5. Run remote klippy and collect output.
- Use task `Run Startup.gcode on Test Rig`.
- This task removes the previous remote `klippy.log`, runs klippy with `startup.gcode`, and then downloads the fresh log locally.
- Capture task exit status and inspect `.github/skills/run-test-rig/klippy.log`.
- Enforce no-command-error rule: fail immediately on `Unknown command:` or traceback.

6. Evaluate with decision points.
- If command unknown: fail run. Update/extend task command list before rerun.
- If startup/init not complete before test command: sequencing issue. Adjust startup gcode order, rerun.
- If expected pattern missing: mark inconclusive unless equivalent evidence exists.
- If forbidden pattern appears (or traceback): fail and report first causal lines.

7. Report with acceptance verdict.
- State pass/fail/inconclusive.
- Provide key evidence lines, not full raw log.
- Include exact next action if inconclusive.

## Decision Points
- Task path:
  - Use `Run Startup.gcode on Test Rig` as the single execution task.
  - Use `Export Startup.gcode to Test Rig` only when you need to push `startup.gcode` without running.

- Re-run criteria:
  - Re-run when logs show command mismatch, startup ordering issue, or stale log contamination.
  - Do not re-run unchanged scenario repeatedly after root cause identified.

- No-error policy:
  - Test task definitions must include checks for `Unknown command:` and traceback signatures.
  - Any run that emits command errors is a failed validation, not a soft warning.

- Evidence threshold:
  - Pass requires positive expected-behavior evidence + no fatal errors.
  - Inconclusive if environment/macro mismatch blocks target path.

## Completion Checklist
- [ ] Code export to rig successful.
- [ ] Scenario script updated in `.github/skills/run-test-rig/startup.gcode`.
- [ ] `Run Startup.gcode on Test Rig` task used to execute scenario and download fresh `klippy.log`.
- [ ] If non-blocking gcodes tested, `startup.gcode` includes `G4 P<ms>` dwell to allow completion.
- [ ] If manual hardware setup required, user was asked to perform setup before execution.
- [ ] No `Unknown command:` lines were emitted.
- [ ] No traceback lines were emitted.
- [ ] Expected evidence present for target scenario.
- [ ] Forbidden signature/traceback absent.
- [ ] Verdict reported: pass/fail/inconclusive.
- [ ] Verdict not inconclusive, or next rerun action documented.

## Reference Command Pattern
- Export: VS Code task `Export Code to Test Rig`.
- Scenario gcode authoring: edit `.github/skills/run-test-rig/startup.gcode`.
- Optional scenario gcode deployment-only: VS Code task `Export Startup.gcode to Test Rig`.
- Run scenario on rig: VS Code task `Run Startup.gcode on Test Rig` — runs `.github/skills/run-test-rig/startup.gcode`, clears old remote log first, downloads fresh local `klippy.log` when done.
- Forbidden: any other direct SSH command usage for test execution.
- Custom run pattern must be implemented as VS Code tasks.
