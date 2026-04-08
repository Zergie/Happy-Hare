## Plan: Fix BLDC MMU_LOAD Toolhead Homing in _trace_bldc_homing_move

Implement a targeted fix for MMU_LOAD where the failure is observed in `_trace_bldc_homing_move` during toolhead-homing with `motor="gear+extruder"`. Scope is strictly `_trace_bldc_homing_move` in `extras/mmu/mmu.py`.

**Steps**
1. Phase 1 - Confirm contracts and failing behavior (blocks all later phases)
1. Verify expected contract for `_trace_bldc_homing_move` with `motor="gear+extruder"`: BLDC and extruder must move concurrently during homing.
2. Confirm observed failure condition: toolhead-homing phase reached, BLDC moves, extruder does not.
3. Keep safety contract: BLDC stop/cleanup on all exits (success, timeout, exception).

2. Phase 2 - Implement focused fix in `_trace_bldc_homing_move` only (depends on Phase 1)
1. Update `_trace_bldc_homing_move` to explicitly command concurrent extruder motion when `motor="gear+extruder"` during homing.
2. Keep existing BLDC start/poll/stop flow and endstop trigger semantics intact.
3. Preserve sensor source fallback behavior (MMU sensor first, rail endstop fallback).
4. Preserve sync-mode restoration and current restoration semantics in `finally` cleanup.

3. Phase 3 - Motion accounting and state coherence (depends on Phase 2)
1. Ensure returned `(actual, homed, measured, delta)` remains coherent with homing stop position.
2. Keep `mmu_toolhead` position updates consistent with actual moved distance.
3. Avoid introducing sequential fallback behavior in combined mode.

4. Phase 4 - Diagnostics quality (parallel with Phase 3)
1. Improve homing logs in `_trace_bldc_homing_move` to show:
- motor mode
- endstop/sensor source
- trigger outcome
- moved distance
- whether combined mode path was used
2. Keep error messages actionable for missing mapping, unreachable endstop, or timeout.

5. Phase 5 - Validation and regression checks (depends on Phases 2-4)
1. Compile-check changed module.
2. Rig checks focused on homing path:
- `MMU_TEST_HOMING_MOVE ENDSTOP=unit_0_mmu_gate MOTOR=gear+extruder`
- `MMU_LOAD` with toolhead sensor present; verify toolhead-homing step advances with extruder motion.
3. Confirm no regression in:
- `MMU_TEST_HOMING_MOVE ENDSTOP=unit_0_mmu_gate MOTOR=gear`
- BLDC stop/cleanup guarantees on timeout/error paths.

**Relevant files**
- `c:/GIT/YAMMU/Firmware/Happy-Hare/extras/mmu/mmu.py` — only file in scope, only `_trace_bldc_homing_move` targeted.

**Verification**
1. `python -m py_compile extras/mmu/mmu.py`
2. Run homing-focused rig tests and inspect `klippy.log` for BLDC+extruder concurrent homing evidence.
3. Baseline current failure signature for `MMU_LOAD`: `Failed to reach toolhead sensor after moving 120.0mm.`
4. Post-fix acceptance: this exact failure line is no longer emitted for the same `MMU_LOAD` scenario, and no replacement traceback appears.
5. Confirm preservation of existing endstop guard behavior.

**Decisions**
- Included: `_trace_bldc_homing_move` fix only.
- Excluded: non-homing `trace_filament_move` branch changes, config/schema changes, Klipper core edits.
- Safety: mandatory BLDC stop and cleanup on all exits remains non-negotiable.
- Compatibility: preserve command interfaces and sync mode semantics.

**Further Considerations**
1. Optional follow-up after this fix: evaluate non-homing `gear+extruder` BLDC path separately if additional issues remain.
