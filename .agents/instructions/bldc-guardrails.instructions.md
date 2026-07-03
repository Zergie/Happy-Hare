---
description: "BLDC implementation guardrails for Happy-Hare: module boundaries, sync-mode behavior, lookup_object contract safety, and klipper/kalico compatibility."
applyTo: "extras/**/*.py"
---

# BLDC Guardrails

Apply these rules for BLDC-related work in Happy-Hare Python files.

## Scope and Ownership
- Keep BLDC control logic in `extras/mmu/mmu_gear_bldc.py`.
- Keep `extras/mmu/mmu.py` focused on integration/orchestration.
- Do not edit Klipper source files for BLDC feature implementation.
- Do not edit Kalico source files for BLDC feature implementation.
- All changes must be compatible with Klipper and Kalico.

## Output Pin Control
- Do not call `set_pwm` or `set_digital` directly during motion.
- Use proper queuing mechanisms to ensure pin writes are synchronized with motion commands.

## Runtime and Naming Contracts
- Treat names passed to `lookup_object(...)` as fixed contracts.
- Validate `lookup_object(...)` identifiers against Klipper/Happy-Hare source definitions before changing related names.
- Preserve public config-facing names unless a compatibility-safe migration is explicitly required.

## Sync and Motion Behavior
- Preserve sync modes: `gear`, `extruder`, `gear+extruder`, `extruder+gear`.
- In `gear+extruder` and `extruder+gear`, BLDC and extruder must move concurrently (not sequentially).

## Multi-MMU Behavior
- Support per-unit BLDC sections (for example `[mmu_gear_bldc unit1]`, `[mmu_gear_bldc unit2]`) with correct gate-range routing.
- Mixed topology must be possible: one MMU unit may use BLDC while another unit uses stepper gear drive in the same configuration.