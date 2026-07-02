# MMU_TEST_MOVE Motor-Mode Reference

Last verified against source: 2026-06-26.

This note describes what each `MOTOR=` mode does in `MMU_TEST_MOVE` and how motion is routed internally.

## Command intent

`MMU_TEST_MOVE` is a low-level filament movement test command. It routes movement through one of four motor modes:

- `gear`
- `extruder`
- `gear+extruder`
- `synced`

Internally, all four go through:

1. `cmd_MMU_TEST_MOVE`
2. `_move_cmd`
3. `trace_filament_move`

`MMU_TEST_MOVE` is wrapped in `wrap_sync_gear_to_extruder()`, so any temporary sync state used during the test is reconciled/restored when the command exits.

## Quick mode matrix

| `MOTOR=` | Primary driver | Internal sync mode | Which steppers move | Typical use |
|---|---|---|---|---|
| `gear` | MMU toolhead Y-axis | `GEAR_ONLY` | Gear rail only | Pure MMU gear-path testing |
| `extruder` | MMU toolhead Y-axis | `EXTRUDER_ONLY_ON_GEAR` | Extruder stepper only (relocated onto MMU rail) | Extruder-only filament motion via MMU motion path |
| `gear+extruder` | MMU toolhead Y-axis | `EXTRUDER_SYNCED_TO_GEAR` | Gear + extruder concurrently on MMU rail | Coupled MMU-driven dual-stepper motion |
| `synced` | Printer toolhead E-axis | `GEAR_SYNCED_TO_EXTRUDER` (aka internal `extruder+gear`) | Extruder drives, gear follows concurrently | Print-like sync behavior (extruder-led) |

## Common behavior for all modes

### 1) Validation and setup

`_move_cmd`:

- accepts only `gear`, `extruder`, `gear+extruder`, `synced`
- reads parameters:
  - `MOVE` (default `100`)
  - `SPEED` (optional, mode-defaulted if omitted)
  - `ACCEL` (optional, mode-defaulted if omitted)
  - `WAIT` (default `1`)
- hidden/debug parameters in `cmd_MMU_TEST_MOVE`:
  - `DEBUG` (default `0`)
  - `ALLOW_BYPASS` (default `0`)

### 2) Grip/release handling

Before motion:

- `MOTOR=extruder` -> `selector.filament_release()`
- all others -> `selector.filament_drive()`

### 3) Per-gate override

After mode defaults are computed, `gate_speed_override` scales both speed and accel.

### 4) BLDC routing

If selected gate uses BLDC and mode is one of `gear`, `gear+extruder`, `synced` (non-homing path), BLDC path is engaged.

---

## Detailed mode behavior

## `MOTOR=gear`

### Motion routing

- `trace_filament_move` sets sync mode to `GEAR_ONLY`.
- MMU rail (`mmu_toolhead.move`) is commanded by distance `MOVE`.
- Extruder is not part of rail motion in this mode.

### Speed defaults

Non-homing selection logic:

- long negative move -> `gear_unload_speed` / `gear_unload_accel`
- long positive move from spool -> `gear_from_spool_speed` / `gear_from_spool_accel`
- long positive move from buffer -> `gear_from_buffer_speed` / `gear_from_buffer_accel`
- short move -> `gear_short_move_speed` / `gear_short_move_accel`

(Threshold is `gear_short_move_threshold`.)

### BLDC specifics

- For BLDC gates, `start_move(dist, speed)` is used.
- BLDC queue includes kick/cruise/stop phases so standalone gear tests keep BLDC active through the move duration.
- `WAIT=0` is explicitly supported for this mode (BLDC controller self-terminates via queued stop phase).

---

## `MOTOR=extruder`

### Motion routing

- `trace_filament_move` sets sync mode to `EXTRUDER_ONLY_ON_GEAR`.
- Extruder stepper is relocated to MMU rail ownership.
- MMU gear steppers are disabled for movement in this mode.
- Command still runs as MMU rail motion (`mmu_toolhead.move`), but only extruder stepper contributes.

### Speed defaults

- positive move -> `extruder_load_speed`
- negative move -> `extruder_unload_speed`
- accel -> `extruder_accel`

### Practical meaning

This is an extruder-only filament move using the MMU motion path and sync infrastructure, not the printer toolhead E planner.

---

## `MOTOR=gear+extruder`

### Motion routing

- `trace_filament_move` sets sync mode to `EXTRUDER_SYNCED_TO_GEAR`.
- Extruder stepper is appended to MMU gear rail.
- A single MMU rail move command drives both gear and extruder steppers together.

### Concurrency property

This mode is concurrent by design: one rail command emits both step streams in the same motion timeline.

### Speed defaults

- positive move -> `extruder_sync_load_speed`
- negative move -> `extruder_sync_unload_speed`
- accel -> `min(max(gear_from_buffer_accel, gear_from_spool_accel), extruder_accel)`

### Practical meaning

Use this when MMU side should lead while extruder follows in lockstep.

---

## `MOTOR=synced`

### Motion routing

- `trace_filament_move` sets sync mode to `GEAR_SYNCED_TO_EXTRUDER`.
- For BLDC, selected gate is explicitly marked as sync-active (`_route_bldc_sync_to_selected_gate(True)`).
- Movement is issued as printer extruder motion (`toolhead.move` on E axis), not MMU rail move.
- Gear follows the extruder through sync linkage.

### Internal naming note

In `MmuToolHead` naming, this state corresponds to internal `"extruder+gear"` (`GEAR_SYNCED_TO_EXTRUDER`).

### Concurrency property

This is also concurrent by design: extruder is driver, gear is follower within the same synced timeline.

### Speed defaults

Same default pair as `gear+extruder`:

- positive move -> `extruder_sync_load_speed`
- negative move -> `extruder_sync_unload_speed`
- accel -> `min(max(gear_from_buffer_accel, gear_from_spool_accel), extruder_accel)`

### Practical meaning

This is the closest `MMU_TEST_MOVE` mode to print-like "gear follows extruder" behavior.

---

## WAIT semantics

`WAIT` controls whether command waits for move queues to complete before returning:

- `WAIT=1` (default): synchronous command completion.
- `WAIT=0`: fire-and-return behavior.

Queue wait helper waits both printer toolhead and MMU toolhead queues.

### Important operational note

For synchronized modes (`gear+extruder`, `synced`), using `WAIT=1` is the safest/most deterministic choice for validation and logging.

For BLDC `MOTOR=gear`, `WAIT=0` is explicitly supported because queued BLDC stop phases can complete after command return.

---

## What this means physically

- `gear`: filament pushed/pulled only by MMU gear drive.
- `extruder`: filament moved only by extruder stepper (MMU gear not driving).
- `gear+extruder`: both motors advance/retract together, MMU-rail-driven.
- `synced`: extruder move leads, MMU gear mirrors/follows in synced mode.

---

## Gotchas

1. `synced` is valid for `MMU_TEST_MOVE`, but not for `MMU_TEST_HOMING_MOVE`.
2. If bypass or disabled checks fail upstream, command exits early.
3. BLDC sync behavior depends on selected gate and sync routing to that gate.
4. Seeing only one side move in a synchronized mode usually indicates sync ownership/routing state mismatch, not command parsing.

---

## Source anchors

- `extras/mmu/mmu.py`
  - `cmd_MMU_TEST_MOVE`
  - `_move_cmd`
  - `trace_filament_move`
  - `movequeues_wait`
  - `_route_bldc_sync_to_selected_gate`
  - `wrap_sync_gear_to_extruder`
- `extras/mmu_machine.py`
  - `MmuToolHead` sync constants
  - `MmuToolHead.sync`
  - `sync_mode_to_string`
- `extras/mmu/mmu_gear_bldc.py`
  - `ProcessMoveSyncMonitor` (`process_move` hook path)
  - `start_move`
  - `queue_trapzoid_move`
  - `set_sync_enabled`, `_handle_synced`, `_handle_unsynced`
- `.github/skills/run-test-rig/tests/no_filament_loaded/test_run_test_rig_bldc.py`
  - scenario coverage for `MOTOR=gear` and `MOTOR=synced`
