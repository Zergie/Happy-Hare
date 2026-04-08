## Plan: Per-Gate BLDC Direction Strategy

Add a backward-compatible, configurable per-gate BLDC direction mapping so one gate can run forward while another runs reversed for the same requested filament move distance. Keep BLDC pin control internals unchanged and implement this feature entirely in `mmu_gear_bldc.py` without changing `mmu.py`.

**Steps**
1. Phase 1 - Confirm Scope and Existing Routing
1. Confirm this feature is only for BLDC gear drive paths and does not change stepper-only behavior.
2. Confirm current routing model: one BLDC instance per unit with gate-to-unit lookup, so per-gate inversion must be applied at call sites where gate and requested distance are known.

2. Phase 2 - Config Design (blocks implementation)
1. Introduce a BLDC direction map in each `[mmu_gear_bldc]` section (0=normal, 1=reversed) so mapping is defined where BLDC hardware is defined.
2. Define strict validation and normalization rules:
- Accept only integer/boolean style values.
- Validate against BLDC section scope using `MmuGearBldc(first_gate, num_gates)` constructor values (local index range `0..num_gates-1`).
- Fail fast clearly for malformed values (hard error; no auto-padding).
3. Keep full backward compatibility by defaulting to all-zero mapping (existing behavior unchanged).
4. Read and validate this configuration in `mmu_gear_bldc.py` (the `[mmu_gear_bldc]` owner) and apply mapping internally before direction is set.

**Configuration example**
```ini
# Unit0 uses BLDC and defines its local map (unit-local gate indexing)
[mmu_gear_bldc unit0]
dir_pin: mmu:YAMMU_ESPOOLER_DIR_0
pwm_pin: !mmu:YAMMU_ESPOOLER_PWM_0
# Unit0 has 4 gates -> 4 entries
bldc_gate_direction_map: 0,1,0,1
```

For this example:
- For BLDC gates in unit0, `MMU_SELECT GATE=0` + `MMU_TEST_MOVE MOVE=100 MOTOR=gear` => BLDC dir pin uses normal forward direction.
- For BLDC gates in unit0, `MMU_SELECT GATE=1` + `MMU_TEST_MOVE MOVE=100 MOTOR=gear` => BLDC dir pin uses reversed direction.
- Gates in non-BLDC units continue using stepper behavior unchanged.

3. Phase 3 - BLDC Mapping Design
1. Add a single helper in `mmu_gear_bldc.py` to translate requested BLDC direction by currently selected gate:
- Input: requested direction or signed distance plus current gate from `self.mmu.gate_selected`.
- Output: effective direction/signed distance after applying `bldc_gate_direction_map`.
2. Use this helper consistently for all BLDC move entry points, including:
- non-homing gear moves
- synchronized gear+extruder BLDC moves
- BLDC homing moves
- any calibration/special BLDC move paths that call BLDC start/run methods
3. Ensure extruder/stepper requested motion semantics remain unchanged; only BLDC direction command is inverted per gate.
4. Add BLDC diagnostics that mirror stepper logging style for testing: include unit scope, global gate index, requested distance, effective BLDC distance after mapping, configured map bit/value, and effective BLDC direction.

4. Phase 4 - Edge-Case and Safety Design
1. Define behavior when no gate is selected (for example keep current behavior with no inversion).
2. Verify mixed gate direction within one BLDC unit works without creating extra queues or BLDC instances.
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
- Confirm run-level traces include deterministic `BLDC_MAP:` diagnostics with fields: `gate`, `requested_dist`, `effective_dist`, `map_value`, `effective_dir`.
- Confirm no Unknown command and no traceback.
4. Regression checks:
- BLDC behavior is unchanged when parameter is absent.
- Encoder-gate and sensor-gate homing still complete.
- Sync behavior in combined modes remains concurrent.

6. Phase 6 - Documentation and Rollout
1. Add config documentation where users tune MMU parameters and BLDC setup.
2. Add one practical example: gate 0 normal, gate 1 reversed.
3. Place config in each relevant `[mmu_gear_bldc]` section near `dir_pin`/`pwm_pin` for discoverability and hardware locality.
4. Note migration behavior: if new parameter omitted, behavior is identical to current releases.

**Relevant files**
- c:/GIT/YAMMU/Firmware/Happy-Hare/extras/mmu/mmu_gear_bldc.py — parse and validate `bldc_gate_direction_map` in `[mmu_gear_bldc]` and expose per-unit mapping.
- c:/GIT/YAMMU/Firmware/printer_data/base/mmu_hardware.cfg — user-facing parameter definition in each `[mmu_gear_bldc]` section.
- c:/GIT/Happy-Hare.wiki/Configuring-mmu_hardware.cfg.md — user documentation update for new setting and example usage.

**Verification**
1. Add a startup scenario that issues the same signed move on two gates with opposite map values and compare effective dir pin behavior in logs.
2. Verify `BLDC_MAP:` logs are emitted in stepper-like style and include mapping details needed to prove per-gate reversal behavior.
3. Run MMU unload and load scenarios on both gates and verify no extruder skip regressions introduced by mapping.
4. Validate BLDC homing path still succeeds for both normal and reversed gates.
5. Validate unchanged behavior when parameter is absent.

**Decisions**
- Included scope: design and implementation strategy for configurable per-gate BLDC direction inversion.
- Excluded scope: Klipper core changes, BLDC queue architecture refactors, and multi-instance-per-gate BLDC redesign.
- Key design choice: apply inversion entirely inside `mmu_gear_bldc.py` so no changes are required in `mmu.py`.

**Further Considerations**
1. Keep per-unit indexing rules explicit in docs for multi-unit users (map indexes are local to each `[mmu_gear_bldc unitX]` scope).
2. Validation policy is hard error on any list-length mismatch or malformed value.
3. Keep `BLDC_MAP:` log token stable for long-term test automation and support diagnostics.
