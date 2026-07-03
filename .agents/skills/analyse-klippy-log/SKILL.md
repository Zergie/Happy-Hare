---
name: analyse-klippy-log
description: "Analyse latest test-rig klippy.log for trigger gcode, fatal errors, BLDC anomalies, and likely root-cause location in Happy-Hare MMU modules."
argument-hint: "Optional focus: unit (default unit0), symptom, or command. Use 'all' for whole-log scan."
user-invocable: true
---

# Analyse klippy.log

## When To Use
- Need post-run diagnosis from test rig `klippy.log`.
- Need likely bug location in BLDC/MMU integration code.
- Need fast pass/fail signal: fatal error, BLDC anomaly, or clean run.

## Outcome
Produce terse diagnostic report with:
- Triggering gcode sequence from recent request dump.
- Fatal error findings with context.
- BLDC anomaly findings and optional PWM timeline.
- Root-cause hypothesis mapped to file + function + minimal fix sketch.

## Inputs
- Primary log: `C:/git/YAMMU/Firmware/Happy-Hare/.agents/skills/run-test-rig/klippy.log`
- Focus arg (`$args`):
  - Default: `unit0`
  - `all`: disable focus filtering in Steps 2-4
  - Any other text: treat as unit/symptom/command filter

## Procedure
1. Validate log exists and non-empty.
- If missing/empty, stop and output exactly:
  `No log found. Run the Get klippy.log from Test Rig VS Code task first.`

2. Scope to most recent startup.
- Find last `===== Config file =====` occurrence.
- Analyse only lines after that marker.
- Ignore lines matching `^Stats \d+\.\d+:`.

3. Extract triggering gcode.
- Find `Dumping 20 requests for client` block near end.
- Extract listed gcode requests.
- If block missing, ask exactly:
  `The Dumping 20 requests for client block is missing. What gcode sequence did you run?`

4. Detect fatal errors (focus-filtered unless `all`).
- Scan for:
  - `Transition to shutdown state`
  - `MCU shutdown`
  - `Timer too close`
  - `Unhandled exception`
  - `Traceback (most recent call last)`
  - `Error`/`error` near BLDC or stepper sections
- Ignore known benign harness noise:
  - `AttributeError: 'Mmu' object has no attribute 'clock_to_print_time'`
  - `BlockingIOError: [Errno 11] Resource temporarily unavailable`
- For each hit, report timestamp/print_time and include ±5 lines context.

5. Detect BLDC anomalies (focus-filtered unless `all`).
- Scan `STEPPER: BLDC_*` lines.
- Flag suspicious patterns:
  - `applied_pwm=0.0000` while motor should spin
  - `source=startup_inhibit` during move
  - `BLDC_SET_PIN: discard`
  - `base_pwm=0.0000` with non-zero commanded speed
  - rapid `applied_pwm` oscillation
  - unbounded `correction_pwm`
  - `BLDC_CONTROL: source=stopped` during active move
  - `BLDC_SET_PIN: kick` with no transition ramp
- For each hit, include relevant lines and unit/gate.
- If one move has >=5 `BLDC_SET_PIN` entries, emit PWM timeline table:

| print_time | applied_pwm | note |
|---|---|---|
| ... | ... | ... |

- Add note only for anomalies (abrupt drop, clamp to zero, non-monotonic ramp).

6. Infer root cause.
- If no anomalies and no fatal errors: output `Log clean` in root-cause section.
- Otherwise provide:
  - one-sentence most likely root cause
  - target file and likely function in:
    - `extras/mmu/mmu_gear_bldc.py`
    - `extras/mmu/mmu_machine.py`
    - `extras/mmu/mmu_sync_controller.py`
    - `extras/mmu/mmu.py`
  - minimal code change sketch
  - if ambiguous, give top two hypotheses ranked by likelihood

## Decision Points
- Missing or empty log: stop early with required message.
- Missing request dump block: ask user for executed gcode sequence.
- Focus application:
  - Apply focus filter only to Steps 4-6.
  - Step 3 always uses whole scoped log region.
- Clean decision:
  - Only declare clean when both fatal-errors and BLDC-anomalies sections are empty.

## Output Contract
- Prose style: caveman mode (terse fragments, no filler).
- Keep log excerpts, code, and tables unchanged.
- Use exact section layout:

```markdown
## Triggering gcode
<list of commands>

## Fatal errors
<findings or "None">

## BLDC anomalies
<findings or "None">

## Root cause
<hypothesis + file + function + fix sketch>
```

## Completion Checklist
- [ ] Checked log exists and is non-empty.
- [ ] Scoped to lines after last `===== Config file =====`.
- [ ] Ignored `Stats` telemetry lines.
- [ ] Extracted triggering gcode or asked required missing-block question.
- [ ] Reported fatal errors with ±5 context.
- [ ] Reported BLDC anomalies with unit/gate context.
- [ ] Added PWM timeline if threshold met.
- [ ] Produced ranked root-cause hypothesis (or `Log clean`).
- [ ] Matched exact output sections and caveman prose.