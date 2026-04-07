## Plan: BLDC Calibration via MMU_CALIBRATE_GEAR

Unify BLDC and stepper gear calibration using the existing `MMU_CALIBRATE_GEAR` command with `rotation_distance` semantics, reusing the existing persistence path (`mmu_vars.cfg` and `calibration_manager.update_gear_rd()`).

**Current State Clarification**
- Current BLDC implementation uses `mm_per_rev` in code and in `[mmu_gear_bldc]` config.
- This plan renames BLDC terminology to `rotation_distance` while preserving the same physical meaning (distance per motor revolution).

**Unified Semantics**
- Both stepper and BLDC gear drives store and calibrate `rotation_distance` (distance moved per motor revolution).
- For BLDC: `rotation_distance` = prior `mm_per_rev` value (directly substitutable).
- Only `MMU_CALIBRATE_GEAR MEASURED=<value>` is acceptable for operator calibration input.
- `mmu_gear_rotation_distances[0]` is calibration for gate 0, `mmu_gear_rotation_distances[1]` is calibration for gate 1, and so on.
- Persistence remains via existing `mmu_vars.cfg` key `mmu_gear_rotation_distances`.
- Mixed BLDC and stepper gear drives in one MMU are not supported.

**Implementation Steps**

**(A) Code Changes**
1. Rename `mm_per_rev` → `rotation_distance` in mmu_gear_bldc.py: Update all field references, method signatures, and internal calculations to use `rotation_distance` terminology consistently with Klipper stepper convention.
2. Update BLDC config loading (mmu_gear_bldc.py): Load per-gate `rotation_distance` from `mmu_vars.cfg` at init, with config fallback only when that gate is uncalibrated.
3. Keep uncalibrated behavior warning-based (stepper parity): do not plan to set a fixed default calibration value in the plan; emit warning when gate calibration is missing and fallback is used.
4. Integrate routing in mmu_calibration_manager.py: detect whether configured gear drive is BLDC or stepper and call the correct setter/path for that type when `update_gear_rd()` is invoked.
5. Enforce unsupported mixed mode at config load: add startup guardrail checks that reject configurations mixing BLDC and stepper gear drives across gates/units during configuration parsing.
6. Add operator messaging: ensure MMU_CALIBRATE_GEAR log output indicates detected gear type (BLDC or stepper), gate index, and saved `rotation_distance`.

**(B) Calibration Procedure (Operator Facing)**

Following the standard MMU_CALIBRATE_GEAR pattern:

1. **Select gate 0:**
   ```
   MMU_SELECT GATE=0
   ```

2. **Get filament visible at MMU exit** (the measurement reference point):
   ```
   MMU_TEST_MOVE MOVE=50
   ```
   (Repeat until filament is visible at the exit point)

3. **Remove bowden tube and cut filament flush** with ECAS connector at MMU output

4. **Move exactly 100mm of filament:**
   ```
   MMU_TEST_MOVE MOVE=100
   ```

5. **Measure the actual length** with a ruler from the ECAS connector to the filament end

6. **Run calibration with measured value** (example: if ruler shows 102.5mm):
   ```
   MMU_CALIBRATE_GEAR MEASURED=102.5
   ```
   The command calculates the correct `rotation_distance` and persists to `mmu_vars.cfg`.

7. **Validate** by cutting filament flush again and running:
   ```
   MMU_TEST_MOVE MOVE=100
   ```
   Should now move exactly 100mm

**(C) Test Rig Validation**

1. Deploy code changes to test rig
2. Restart Klipper
3. Run calibration procedure steps (B) above with test filament
4. Verify `mmu_vars.cfg` shows updated `mmu_gear_rotation_distances[0]` for gate 0 calibration
5. Execute synced move to confirm BLDC operates correctly with calibrated value:
   ```
   MMU_TEST_MOVE MOVE=400 SPEED=200 MOTOR=synced
   ```
6. Restart Klipper again to confirm `rotation_distance` loads correctly from `mmu_vars.cfg`
7. Verify mixed BLDC+stepper configuration is rejected during config load with a clear startup error/warning

**Relevant files**
- c:/GIT/YAMMU/Firmware/Happy-Hare/extras/mmu/mmu_gear_bldc.py — rename all `mm_per_rev` references to `rotation_distance`; update parser/setters; apply per-gate load with warning fallback.
- c:/GIT/YAMMU/Firmware/Happy-Hare/extras/mmu/mmu_calibration_manager.py — detect BLDC vs stepper gear drive and route `update_gear_rd()` to correct runtime sink.
- c:/GIT/YAMMU/Firmware/Happy-Hare/extras/mmu/mmu.py — ensure MMU_CALIBRATE_GEAR path remains `MEASURED=` workflow and emits gear-type-aware messaging.
- c:/GIT/YAMMU/Firmware/printer_data/base/mmu_hardware.cfg — rename BLDC field `mm_per_rev` to `rotation_distance` (fallback only, not calibrated source of truth).
- c:/GIT/YAMMU/Firmware/printer_data/mmu_vars.cfg — persistence destination for `mmu_gear_rotation_distances` list.

**Verification**
1. Deploy code changes to test rig and restart Klipper (clean startup with no errors)
2. Run full calibration procedure (section B) with test filament:
   - Confirm `MMU_CALIBRATE_GEAR MEASURED=<value>` calculates and persists new `rotation_distance`
   - Verify log output shows detected gear type (BLDC or stepper) and calibrated gate index
3. Validate move accuracy:
   - Cut filament flush and run `MMU_TEST_MOVE MOVE=100`
   - Measure output; should be 100mm (or within accepted tolerance)
4. Restart Klipper and confirm:
   - BLDC loads persisted `rotation_distance` value from `mmu_vars.cfg`
   - Synced move operates correctly: `MMU_TEST_MOVE MOVE=400 SPEED=200 MOTOR=synced`
5. Guardrail checks:
   - Existing stepper-only MMU calibration still works unchanged
   - Mixed BLDC + stepper MMU configuration is rejected as unsupported during config load/startup (not at runtime command execution)

**Decisions**
- Full rename: standardize on `rotation_distance` throughout MmuGearBldc class, mmu_hardware.cfg, and all setter/getter methods. No internal aliases.
- Unify BLDC and stepper calibration: both use `MMU_CALIBRATE_GEAR MEASURED=` procedure.
- Reuse existing persistence: `calibration_manager.update_gear_rd()` handles gear-type detection and routes to the correct runtime sink.
- Procedural parity: BLDC calibration follows same measurement workflow as stepper.
- No fixed calibrated default in plan: missing calibration should warn and use fallback behavior.
- Mixed BLDC and stepper drives are explicitly unsupported and rejected only during config load/startup.
