---
name: "analyse klippy.log"
description: "Analyse klippy.log after a test rig run. Finds triggering gcode, checks for exceptions, Timer too close, and BLDC misbehaviour. Points to bug location in mmu_gear_bldc.py, mmu_sync_controller.py, or mmu.py — or declares log clean."
argument-hint: "Optional: unit to focus on (default: unit0) or specific symptom/command"
agent: "ask"
---

Analyse `C:/git/YAMMU/Firmware/Happy-Hare/.github/skills/run-test-rig/klippy.log` from the most recent test rig run.

**Focus**: `$args` (if not provided, default to `unit0`). Restrict Steps 2–4 to log lines matching that unit or symptom unless the argument is `all`.

If `C:/git/YAMMU/Firmware/Happy-Hare/.github/skills/run-test-rig/klippy.log` is missing or empty, stop immediately and say: "No log found. Run the `Get klippy.log from Test Rig` VS Code task first."

## Step 0 — Scope the log

Before anything else:
1. Find the **last occurrence** of `===== Config file =====` in the log. This marks the most recent Klipper startup.
2. Analyse **only lines after that marker** for all subsequent steps.
3. **Ignore all lines matching** `^Stats \d+\.\d+:` — they are periodic telemetry and add no diagnostic value.

## Step 1 — Find triggering gcode

Locate the `Dumping 20 requests for client` block near the end of the log.
Extract the gcode requests listed there to establish what triggered the failure or the sequence under test.

If the block is **not present** in the log, ask the user: "The `Dumping 20 requests for client` block is missing. What gcode sequence did you run?"

## Step 2 — Check for fatal errors

Scan the log for:
- `Transition to shutdown state`
- `MCU shutdown`
- `Timer too close`
- `Unhandled exception`
- Python tracebacks (`Traceback (most recent call last)`)
- Any line containing `Error` or `error` near a BLDC or stepper section

Report each hit with:
- Timestamp / print_time
- The full error line(s) and surrounding context (±5 lines)

## Step 3 — Check BLDC behaviour

Scan `STEPPER: BLDC_*` log lines and flag any of the following as suspicious:

| Pattern | Why suspicious |
|---|---|
| `applied_pwm=0.0000` while motor should be spinning | PWM clamped to zero unexpectedly |
| `source=startup_inhibit` during a move | Tachometer inhibit still active too late |
| `BLDC_SET_PIN: discard` | Pin write discarded — possible timing issue |
| `base_pwm=0.0000` with non-zero commanded speed | Calibration map or RPM-to-PWM issue |
| Rapid oscillation in `applied_pwm` values | PID instability |
| `correction_pwm` growing unbounded | Integral wind-up |
| `BLDC_CONTROL: source=stopped` during active move | Premature stop |
| `BLDC_SET_PIN: kick` with no follow-up speed ramp | Kick-start not transitioning |

For each hit, show the relevant log lines and the gate / unit involved.

If there are 5 or more `BLDC_SET_PIN` log entries for a single move, emit a PWM timeline table under **BLDC anomalies**:

| print_time | applied_pwm | note |
|---|---|---|
| 45326.285 | 0.9999 | |
| 45326.306 | 0.9990 | |

Add a `note` only for anomalous entries (abrupt drop, clamp to 0, non-monotonic ramp).

## Step 4 — Root cause hypothesis

Based on findings from steps 1–3:

- If no anomalies found: state "Log clean" and stop.
- Otherwise:
  1. State the most likely root cause in one sentence.
  2. Point to the specific file and function in [mmu_gear_bldc.py](./extras/mmu/mmu_gear_bldc.py), [mmu_machine.py](./extras/mmu/mmu_machine.py) [mmu_sync_controller.py](./extras/mmu/mmu_sync_controller.py), or [mmu.py](./extras/mmu/mmu.py) where the bug likely lives.
  3. Suggest the minimal code change to fix it.
  4. If evidence is ambiguous, list two hypotheses ranked by likelihood.

## Output format

Use caveman mode for all prose (terse, no filler, fragments OK). Keep log excerpts, tables, and code unchanged.

```
## Triggering gcode
<list of commands>

## Fatal errors
<findings or "None">

## BLDC anomalies
<findings or "None">

## Root cause
<hypothesis + file + function + fix sketch>
```
