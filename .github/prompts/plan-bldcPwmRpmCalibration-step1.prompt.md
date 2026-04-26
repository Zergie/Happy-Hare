## Plan: BLDC PWM-RPM Calibration Pipeline

Goal: add BLDC calibration flow to build PWM->RPM map, persist in mmu_vars.cfg, use mapped conversion first, fallback to current linear _rpm_to_pwm when map or tachometer not usable.

**Implementation Phases**
1. Data model + guards
1. Add persisted-variable key `mmu_bldc_map` in extras/mmu/mmu.py for BLDC calibration map. Keep existing lookup_object names and public command contracts unchanged.
2. Define per-unit map schema in extras/mmu/mmu_gear_bldc.py: sorted PWM points, min point count, dedupe by averaging duplicates, range validation.
3. Define runtime guard: invalid/empty map or missing tachometer -> linear _rpm_to_pwm.

2. Command surface
1. Register MMU_CALIBRATE_BLDC in extras/mmu/mmu.py with help text near other calibration commands.
2. Handler resolves active BLDC unit via existing routing; fail fast with explicit message if BLDC or tachometer unavailable.
3. Add MmuGearBldc calibration entry: fixed-step PWM sweep, fixed settle/read windows, tach RPM sampling (valid if RPM > 0 and sample age <= 500 ms), nearest-lookup table build, payload return.
4. If calibration returns fewer than minimum valid points: fail command and do not persist map.

3. Persistence + startup load
1. Save per-unit map via existing save_variable/write_variables path in extras/mmu/mmu.py. No new backend.
2. Load map during startup/ready lifecycle, inject into each MmuGearBldc before motion.
3. Validate on load (type/shape/range). On invalid data, warn and fallback linear.

4. Runtime hookup
1. Route all BLDC speed conversions through one method:
- mapped nearest lookup when calibration valid
- linear _rpm_to_pwm when map invalid/missing
 - clamp out-of-range RPM queries to nearest endpoint PWM
 - on nearest-distance tie, choose higher PWM
2. Apply across normal moves, sync/process_move path, and load/unload/home flows that call set_speed/start_move.
3. Preserve contracts: queue-only pin writes (no direct set_pwm/set_digital outside queue callback), no sync-mode behavior change, combined modes stay concurrent.

5. Observability + docs
1. Extend status/logs: mapping mode, map size, fallback reason.
2. Update docs/config comments listing calibration commands and persistence behavior.

**Target Files**
- Happy-Hare/extras/mmu/mmu.py: command registration/handler, persisted key, startup load wiring.
- Happy-Hare/extras/mmu/mmu_gear_bldc.py: calibration routine, schema validation, mapped conversion + fallback.
- Happy-Hare/extras/mmu/mmu_calibration_manager.py: optional coordinator if calibration flow centralized.
- Happy-Hare/config/base/mmu_sequence.cfg: no direct macro change expected.
- Happy-Hare/.github/skills/run-test-rig/startup.gcode: optional repeatable hardware validation.

**Verification**
1. Calibration command: run MMU_CALIBRATE_BLDC on BLDC+tach rig; confirm sampled points logged.
2. Persistence: confirm per-unit map key in mmu_vars.cfg survives FIRMWARE_RESTART.
3. Runtime: run load/unload/home; confirm mapped mode logs when valid map exists.
4. Fallback: force invalid/missing map; confirm explicit linear fallback log and successful motion.
5. Sync safety: confirm gear+extruder and extruder+gear still concurrent; confirm no direct pin writes outside queue callback.
6. Regression: non-BLDC behavior unchanged; mixed BLDC+stepper routing correct.
7. Hardware logs: inspect klippy.log for map load/use/fallback and no timing/pin errors.

**Validation Tests To Add**
1. Unit test: map schema validation accepts sorted valid points and rejects bad shape/type/range/min-count cases.
2. Unit test: dedupe policy keeps deterministic point set and order.
3. Unit test: conversion selector uses mapped lookup when map valid, linear _rpm_to_pwm when map invalid/missing.
4. Unit test: nearest-point lookup picks expected PWM at low/mid/high RPM query boundaries.
5. Integration test: MMU_CALIBRATE_BLDC handler errors clearly when active unit is not BLDC or tachometer missing.
6. Integration test: persisted per-unit map written through save_variable/write_variables and loaded on startup before first BLDC move.
7. Integration test: invalid persisted payload logs warning and triggers linear fallback without crashing motion flow.
8. Integration test: mixed topology routing (BLDC unit + stepper unit) applies calibration only to BLDC unit.
9. Log/assertion test: combined sync modes (gear+extruder, extruder+gear) remain concurrent after hookup.

**Validation Pass Criteria**
1. Authority metric: unit/integration test pass status in Happy-Hare test runner.
2. All new unit/integration tests pass in Happy-Hare test runner.
3. Calibration-failure path (insufficient valid points) explicitly fails and does not persist map.
4. Out-of-range RPM queries clamp to endpoint PWM and nearest-lookup tie chooses higher PWM.
5. Tach validity gate enforced (RPM > 0 and sample age <= 500 ms) in calibration/runtime paths.
6. At least one failing-path assertion per fallback branch (missing map, invalid map, missing tach).
7. No new direct BLDC pin writes outside queue callback in changed code paths.
8. No regression failures in existing non-BLDC tests.

**Decisions**
- Trigger: MMU_CALIBRATE_BLDC.
- Scope: per-BLDC-unit map, not per gate.
- Runtime model: nearest-point lookup.
- Persisted key: mmu_bldc_map.
- Dedupe policy: average duplicates.
- Clamp policy: clamp to nearest endpoint PWM.
- Nearest tie-break: choose higher PWM.
- Tach validity gate: RPM > 0 and sample age <= 500 ms.
- Calibration failure policy: fail command, do not persist map.
- Missing tach/map: linear fallback.
- Persistence: mmu_vars.cfg via save_variables.
- In scope: BLDC calibration + usage across BLDC moves.
- Out of scope: mmu_sync_controller behavior changes, Klipper core edits, non-BLDC motion redesign.

**Open Detail**
1. Fixed sweep selected; exact numeric sweep parameters (start/end PWM, step count, settle/read windows) still to be set.
2. Future: piecewise interpolation.
3. Future: map reset/export commands.