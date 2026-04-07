## Plan: BLDC Homing and Load Support

Add BLDC-specific homing handling inside movement orchestration so MMU_TEST_HOMING_MOVE and MMU_LOAD work with BLDC, while preserving queue safety and concurrent sync behavior.

**Steps**
1. Phase 1 - Baseline and contracts (blocks all later phases)
1. Keep existing guardrails intact:
- queued pin writes only through BLDC queue callbacks
- one queue per MCU in self.gcrqs
- no Klipper core edits
- preserve lookup_object contracts and sync mode semantics
2. Confirm command scope:
- MMU_TEST_HOMING_MOVE support for gear and gear+extruder with BLDC
- MMU_LOAD support via the same BLDC homing-capable movement path

2. Phase 2 - BLDC homing backend in movement orchestration (depends on Phase 1)
1. In extras/mmu/mmu.py, replace the BLDC homing rejection in trace_filament_move with a BLDC-specific homing branch.
2. Add a dedicated helper in extras/mmu/mmu.py (for example _trace_bldc_homing_move) that:
- resolves mapped endstop names like the current path
- starts BLDC movement asynchronously
- polls sensor state via sensor_manager with bounded timeout from distance/speed
- stops BLDC on trigger or timeout
- returns actual/homed/measured/delta in current trace_filament_move contract format
3. Keep gear+extruder homing concurrent, not sequential.
4. Leave non-BLDC HomingMove path unchanged.

3. Phase 3 - MMU_LOAD integration and collision path hardening (depends on Phase 2)
1. Reuse new BLDC homing behavior through existing _load_gate and _home_to_extruder calls in extras/mmu/mmu.py without interface changes.
2. Update _home_to_extruder_collision_detection in extras/mmu/mmu.py to avoid stepper-only assumptions when BLDC is active, using BLDC-compatible movement plus encoder/sensor feedback.
3. Preserve filament state transitions and gate status semantics.

4. Phase 4 - Diagnostics quality (parallel with Phase 3)
1. Improve homing logs to include BLDC mode, sensor name, trigger outcome, moved distance.
2. Ensure BLDC homing errors are actionable (invalid endstop mapping, missing sensor, timeout).

5. Phase 5 - Validation and regression checks (depends on Phases 2-4)
1. Run diagnostics/compile checks on changed modules.
2. Validate MMU_TEST_HOMING_MOVE on rig in BLDC:
- MOTOR=gear
- MOTOR=gear+extruder
3. Validate MMU_LOAD end-to-end on BLDC:
- gate phase
- bowden phase
- extruder homing phase
- loaded state reached
4. Regression check stepper-only path still uses HomingMove and behaves unchanged.
5. Confirm combined sync modes remain concurrent.

**Relevant files**
- extras/mmu/mmu.py — main orchestration changes for homing branch, helper, load-path reuse, collision path adjustment.
- extras/mmu/mmu_gear_bldc.py — BLDC start/stop reuse, optional homing polling tunables.
- extras/mmu/mmu_sensor_manager.py — sensor/endstop mapping used by BLDC homing checks.
- extras/mmu/mmu_sync_controller.py — verify no concurrency regression.
- printer_data/base/mmu_hardware.cfg — optional tunable additions if needed.

**Verification**
1. Static checks on changed files.
2. Rig commands:
- MMU_TEST_HOMING_MOVE ENDSTOP=unit_0_mmu_gate MOTOR=gear
- MMU_TEST_HOMING_MOVE ENDSTOP=unit_0_mmu_gate MOTOR=gear+extruder
3. MMU_LOAD with BLDC from unloaded state and confirm successful load completion.
4. Stepper-only regression run for MMU_TEST_HOMING_MOVE.
5. Confirm no sequential fallback in combined sync modes.

**Decisions**
- Include BLDC support for MMU_TEST_HOMING_MOVE in gear and gear+extruder modes.
- Polling-based stop overshoot is acceptable for BLDC homing.
- Non-BLDC homing path remains isolated and unchanged.
- No Klipper core or config contract migrations in this scope.

**Further considerations**
1. Optionally expose homing polling interval in BLDC config for tighter high-speed stop behavior.
2. Optionally require consecutive sensor hits to reduce false-positive triggers on noisy sensors.
3. Optionally report last BLDC homing result in status output for easier troubleshooting.
