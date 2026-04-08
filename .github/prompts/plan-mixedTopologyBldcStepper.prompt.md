## Plan: Mixed Topology BLDC + Stepper Units

Enable configurations where one MMU unit uses BLDC gear drive and another unit uses stepper gear drive in the same printer setup.

**Steps**
1. Phase 1 - Current Limitation Audit
1. Identify global mutual-exclusion checks that currently block simultaneous BLDC and stepper definitions.
2. Confirm all runtime branching points that currently assume global drive type.

2. Phase 2 - Per-Unit Drive Resolution Design
1. Resolve drive type per unit (BLDC or stepper) using unit-scoped configuration.
2. Validate each unit has exactly one effective drive type.
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

6. Phase 6 - Docs
1. Add hardware config examples for mixed topology.
2. Document per-unit constraints and troubleshooting.

**Relevant files**
- c:/GIT/YAMMU/Firmware/Happy-Hare/extras/mmu_machine.py — per-unit drive-type resolution and config validation.
- c:/GIT/YAMMU/Firmware/Happy-Hare/extras/mmu/mmu.py — gate-to-drive routing behavior.
- c:/GIT/YAMMU/Firmware/Happy-Hare/extras/mmu/mmu_gear_bldc.py — BLDC path behavior for BLDC units.
- c:/GIT/YAMMU/Firmware/printer_data/base/mmu_hardware.cfg — mixed-topology configuration examples.
- c:/GIT/Happy-Hare.wiki/Configuring-mmu_hardware.cfg.md — documentation update.

**Decisions**
- Included scope: mixed BLDC + stepper unit operation in one configuration.
- Excluded scope: per-gate BLDC direction-map feature details (covered in separate plan).
