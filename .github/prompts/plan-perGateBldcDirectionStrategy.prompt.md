## Plan: Per-Gate BLDC Direction Strategy

Add a backward-compatible, configurable per-gate BLDC direction mapping so one gate can run forward while another runs reversed for the same requested filament move distance. Keep BLDC pin control internals unchanged and implement direction mapping in MMU orchestration where gate context already exists.

**Steps**
1. Phase 1 - Confirm Scope and Existing Routing
1. Confirm this feature is only for BLDC gear drive paths and does not change stepper-only behavior.
2. Confirm current routing model: one BLDC instance per unit with gate-to-unit lookup, so per-gate inversion must be applied at call sites where gate and requested distance are known.

2. Phase 2 - Config Design (blocks implementation)
1. Introduce a new per-gate config list in MMU config parsing (for example a list matching gate count where 0=normal and 1=reversed).
2. Define strict validation and normalization rules:
- Accept only integer/boolean style values.
- Normalize/pad to number of gates with default 0.
- Fail fast or warn clearly for malformed values.
3. Keep full backward compatibility by defaulting to all-zero mapping (existing behavior unchanged).

**Configuration example**
```ini
[mmu]
# Per-gate BLDC direction map: 0=normal, 1=reversed
# Gate 0 -> normal (forward), Gate 1 -> reversed (backward), Gate 2 -> normal, Gate 3 -> reversed
gate_bldc_direction_invert: 0,1,0,1
```

For this example:
- `MMU_SELECT GATE=0` + `MMU_TEST_MOVE MOVE=100 MOTOR=gear` => BLDC dir pin uses normal forward direction.
- `MMU_SELECT GATE=1` + `MMU_TEST_MOVE MOVE=100 MOTOR=gear` => BLDC dir pin uses reversed direction.

3. Phase 3 - Orchestration Mapping Design
1. Add a single helper in MMU orchestrator to translate requested BLDC distance by gate:
- Input: gate index and requested dist.
- Output: same dist or negated dist according to config map.
2. Use this helper consistently for all BLDC move entry points, including:
- non-homing gear moves
- synchronized gear+extruder BLDC moves
- BLDC homing moves
- any calibration/special BLDC move paths that call BLDC start/run methods
3. Ensure extruder/stepper requested motion semantics remain unchanged; only BLDC direction command is inverted per gate.

4. Phase 4 - Edge-Case and Safety Design
1. Define behavior when no gate is selected (for example keep current behavior with no inversion).
2. Verify mixed gate direction within one unit works without creating extra queues or BLDC instances.
3. Preserve direction deadtime and queued pin write behavior in BLDC module (no direct pin-write path changes).
4. Verify sync-mode contracts remain concurrent for combined modes.

5. Phase 5 - Verification Plan
1. Static validation:
- Confirm all BLDC move call sites pass through direction mapping helper.
- Confirm no lookup_object contracts changed.
2. Functional matrix on rig with at least two gates configured opposite directions:
- Gate 0 normal: positive move drives original dir pin state.
- Gate 1 reversed: same positive move drives opposite dir pin state.
- Run load, unload, and homing variants on both gates.
3. Log verification:
- Confirm run-level traces indicate expected gate and effective BLDC direction.
- Confirm no Unknown command and no traceback.
4. Regression checks:
- Stepper-only systems unaffected.
- Encoder-gate and sensor-gate homing still complete.
- Sync behavior in combined modes remains concurrent.

6. Phase 6 - Documentation and Rollout
1. Add config documentation where users tune MMU parameters and BLDC setup.
2. Add one practical example: gate 0 normal, gate 1 reversed.
3. Note migration behavior: if new parameter omitted, behavior is identical to current releases.

**Relevant files**
- c:/GIT/YAMMU/Firmware/Happy-Hare/extras/mmu/mmu.py — gate-aware BLDC orchestration, config parsing, and BLDC call sites.
- c:/GIT/YAMMU/Firmware/Happy-Hare/extras/mmu/mmu_gear_bldc.py — BLDC runtime/pin queue behavior (reference only; keep mostly unchanged).
- c:/GIT/YAMMU/Firmware/printer_data/base/mmu_parameters.cfg — user-facing parameter definition for per-gate BLDC direction map.
- c:/GIT/Happy-Hare.wiki/Configuring-mmu_parameters.cfg.md — user documentation update for new setting and example usage.

**Verification**
1. Add a startup scenario that issues the same signed move on two gates with opposite map values and compare effective dir pin behavior in logs.
2. Run MMU unload and load scenarios on both gates and verify no extruder skip regressions introduced by mapping.
3. Validate BLDC homing path still succeeds for both normal and reversed gates.
4. Validate unchanged behavior when parameter is absent.

**Decisions**
- Included scope: design and implementation strategy for configurable per-gate BLDC direction inversion.
- Excluded scope: Klipper core changes, BLDC queue architecture refactors, and multi-instance-per-gate BLDC redesign.
- Key design choice: apply inversion in mmu orchestrator (gate-aware layer) rather than inside low-level BLDC pin module.

**Further Considerations**
1. Decide parameter naming now for long-term consistency with existing per-gate lists: Option A use a new BLDC-specific list, Option B reuse/extend existing gate-direction semantics.
2. Decide validation strictness: Option A hard error on list length mismatch, Option B auto-pad with warnings for easier migration.
3. Decide whether to add an explicit debug field showing effective BLDC direction per move to simplify support diagnostics.
