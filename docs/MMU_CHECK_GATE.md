# MMU_CHECK_GATE Movement Reference

Last verified against source: 2026-06-26.

This note describes what filament motion should happen when running `MMU_CHECK_GATE` in Happy Hare.

## Command intent

`MMU_CHECK_GATE` verifies one or more gates by:
1. Selecting each target gate.
2. Trying to load filament from that gate into the MMU gate-homing reference point.
3. Parking filament back out of the gate.
4. Marking gate status as available or empty.

If filament was already loaded before the command, it first performs a full unload of the currently loaded tool, then restores the previous tool selection/state when checks finish.

## Target selection rules (priority order)

Only one selection mode is applied (first match wins):
1. `GATE=<n>`
2. `TOOL=<n>` (mapped through `ttg_map`)
3. `ALL=1` (all gates)
4. `GATES=a,b,c`
5. `TOOLS=a,b,c`
6. No params: current selected gate

Special case:
- `TOOLS=` with an empty value (for example `MMU_CHECK_GATE TOOLS=`) exits early with no checks.

## High-level movement timeline

### A) Optional pre-check unload

If filament position is not `UNLOADED`:
1. Command parks toolhead and runs full unload sequence (`_unload_tool(form_tip=FORM_TIP_STANDALONE)`).
2. Filament should end parked at MMU gate-side unload position (not in extruder/toolhead path).

Practical expectation:
- You should see a complete retract path out of hotend/extruder/bowden as configured.
- Gate check loop starts only after this unload completes.

### B) Per-gate check loop

For each target gate:
1. `select_gate(gate)` aligns selector.
2. `_load_gate(allow_retry=False)` tries to pick up and home filament from that gate.
3. On success: gate is marked available.
4. `_unload_gate()` reverse-homes and parks filament back out.
5. Encoder is re-initialized.

If load or unload fails:
- Gate is marked `EMPTY`.
- Filament state forced to `UNLOADED`.
- Gear position reset to 0.

### C) Post-check restore

When not printing (`is_printing() == False`):
- Restores previous tool selection.
- If filament was loaded before command, performs full re-load of that prior tool.

When printing (`is_printing() == True`):
- No post-check restore is performed.
- This command path is intended as a pre-load/pre-print validation flow.

Other runtime nuance:
- On a gate failure while `is_in_print()` is true, the command raises immediately instead of continuing remaining gates.

## Detailed filament motion in one gate check

The exact shape depends on gate homing method.

### Case 1: Gate homing via encoder (`gate_homing_endstop == encoder`)

Load phase (`_load_gate`):
1. Forward gear move up to `gate_homing_max`.
2. Encoder must report movement greater than about 6 mm.
3. If encoder movement is sufficient, gate is considered available and load phase succeeds.

Unload phase (`_unload_gate`):
1. Reverse in steps (`encoder_move_step_size`) until encoder indicates filament cleared.
2. Perform final reverse parking move.
3. Set filament state to `UNLOADED` and zero MMU gear axis reference.

What this should look like:
- A forward pick-up stroke.
- A reverse withdrawal with step-like clearance behavior.
- A short final reverse park stroke.

### Case 2: Gate homing via gate/gear sensor (`mmu_gate` or `mmu_gear_n`)

Load phase (`_load_gate`):
1. Homing move toward sensor (normally positive direction).
2. If already triggered and parking config implies reverse-home, direction may invert.
3. Success when sensor reaches target homing state.

Unload phase (`_unload_gate`):
1. Reverse-home off sensor up to `gate_homing_max`.
2. Final parking reverse move by `gate_parking_distance`.
3. Set filament state to `UNLOADED` and zero MMU gear axis reference.

What this should look like:
- Controlled forward move until sensor trip.
- Controlled reverse move until sensor clears.
- Extra reverse park distance to final parked point.

## BLDC-specific behavior during MMU_CHECK_GATE

For gates using BLDC:
- Non-homing gear moves run through BLDC move start/stop path.
- Homing moves (`homing_move != 0`) use BLDC sensor polling loop (`trace_bldc_filament_move`) instead of Klipper `HomingMove` for the gear leg.
- For BLDC gear homing success, controller uses brake-to-stop behavior after trigger.

Operationally, expected motion still follows the same logical pattern:
1. Move toward detect point.
2. Detect trigger/clear via configured sensor.
3. Stop and park.

## Status outcomes you should expect

Per gate:
- Success path: gate marked available; filament parked out after check.
- Failure path: gate marked empty; filament state forced unloaded; gear axis reset.

Whole command:
- If multiple gates, each gate is checked independently.
- If one gate fails while not printing, process continues and reports that gate empty.
- If in print context and required gate fails, command raises MMU error.

## Key parameters

- `GATE=<n>`: check one gate.
- `TOOL=<n>`: map tool to gate via `ttg_map`, then check that gate.
- `ALL=1`: check all gates.
- `GATES=a,b,c`: check explicit gate list.
- `TOOLS=a,b,c`: check gate(s) mapped from tool list.
- `QUIET=1`: suppress final visual-state summary output.

## Quick physical sanity checklist

After `MMU_CHECK_GATE` finishes:
1. No filament should remain advanced in checked gate path; each checked gate should be parked out.
2. Gate sensor should not remain unexpectedly triggered for a gate that was parked.
3. If starting state had loaded filament and not printing, previous tool/load state should be restored.
4. Gear position reference should be reset after unload parking in each check cycle.

## Source anchors

- `extras/mmu/mmu.py`:
  - `cmd_MMU_CHECK_GATE`
  - `_load_gate`
  - `_unload_gate`
  - `trace_filament_move`
  - `trace_bldc_filament_move`
  - `wrap_sync_gear_to_extruder`
  - `_wrap_suspend_filament_monitoring`
  - `_wrap_suspendwrite_variables`
