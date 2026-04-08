## Plan: Mixed Topology BLDC + Stepper Units

Enable configurations where one MMU unit uses BLDC gear drive and another unit uses stepper gear drive in the same printer setup.

**Steps**
1. Phase 1 - Current Limitation Audit
1. Identify global mutual-exclusion checks that currently block simultaneous BLDC and stepper definitions.
2. Confirm all runtime branching points that currently assume global drive type.

2. Phase 2 - Per-Unit Drive Resolution Design
1. Resolve drive type per unit (BLDC or stepper) using unit-scoped configuration.
2. Validate each unit has exactly one effective drive type.
3. Raise a hard configuration error when any unit resolves to both BLDC and stepper gear definitions.
3. Explicitly replace global `use_bldc`-style decisions with per-unit drive decisions at design level.
4. Keep existing single-type configurations fully backward compatible.

3. Phase 3 - Runtime Integration
1. Route gear motion by gate to the owning unit’s drive implementation.
2. Replace runtime global drive branching with per-unit branching at each relevant call path.
3. Ensure BLDC-only behavior remains unchanged for BLDC units.
4. Ensure stepper-only behavior remains unchanged for stepper units.

4. Phase 4 - Safety and Contracts
1. Preserve queue contracts and BLDC pin-write constraints.
2. Preserve sync-mode concurrency behavior in combined modes.
3. Ensure no lookup_object contract names are changed.

5. Phase 5 - Verification
1. Mixed setup test: unit0 BLDC + unit1 stepper.
2. Per-unit load/unload/homing on representative gates.
3. Confirm no Unknown command, no traceback.
4. Confirm BLDC diagnostics appear only on BLDC units.
5. Confirm stepper diagnostics and behavior are unchanged on stepper units.
6. Confirm BLDC-only setup remains behaviorally identical to baseline.
7. Confirm stepper-only setup remains behaviorally identical to baseline.

6. Phase 6 - Docs
1. Add hardware config examples for mixed topology.
2. Document `num_gates` and unit naming rules clearly:
- `num_gates` in `[mmu_machine]` defines per-unit gate counts as a comma-separated list.
- Unit names are zero-based and map directly to suffix naming, e.g. `unit0`, `unit1`, `unit2`.
- Example: `num_gates: 4,4,2` means `unit0` owns gates 0..3, `unit1` owns gates 4..7, `unit2` owns gates 8..9.
3. Document per-unit constraints and troubleshooting.

**Relevant files**
- Happy-Hare/extras/mmu_machine.py — per-unit drive-type resolution and config validation.
- Happy-Hare/extras/mmu/mmu.py — gate-to-drive routing behavior.
- Happy-Hare/extras/mmu/mmu_gear_bldc.py — BLDC path behavior for BLDC units.
- printer_data/base/mmu_hardware.cfg — mixed-topology configuration examples.
- Happy-Hare.wiki/Configuring-mmu_hardware.cfg.md — documentation update.

**Decisions**
- Included scope: mixed BLDC + stepper unit operation in one configuration.
- Backward-compatibility claim: BLDC-only and stepper-only setups must remain behaviorally unchanged.
- Excluded scope: per-gate BLDC direction-map feature details (covered in separate plan).

**Audit Checklist**
1. Global drive assumptions audited and replaced with per-unit decisions.
2. Any unit with both BLDC and stepper gear definitions raises a deterministic config error.
3. Any unit with neither BLDC nor stepper gear definition raises a deterministic config error.
4. Unit-to-gate ownership from `num_gates` is verified against runtime routing.
5. Mixed setup verified: at least one BLDC unit and one stepper unit in same configuration.
6. BLDC unit behavior unchanged from baseline when stepper unit is present.
7. Stepper unit behavior unchanged from baseline when BLDC unit is present.
8. No `Unknown command` and no traceback in test runs.
