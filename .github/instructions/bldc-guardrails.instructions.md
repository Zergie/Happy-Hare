---
description: "BLDC implementation guardrails for Happy-Hare: module boundaries, pin-write queue rules, sync-mode behavior, and lookup_object contract safety."
applyTo: "extras/**/*.py"
---

# BLDC Guardrails

Apply these rules for BLDC-related work in Happy-Hare Python files.

## Scope and Ownership
- Keep BLDC control logic in `extras/mmu/mmu_gear_bldc.py`.
- Keep `extras/mmu/mmu.py` focused on integration/orchestration.
- Do not edit Klipper source files for BLDC feature implementation.

## Output Pin and Queue Rules
- Use queued pin writes only for BLDC pwm/digital_out control paths.
- Do not call `set_pwm` or `set_digital` outside queue callback functions.
- Use `output_pin.GCodeRequestQueue` when available.
- Fallback to `GCodeRequestQueue` from `extras/mmu_espooler.py` when needed.
- Name queue map `self.gcrqs`.
- Allow exactly one `GCodeRequestQueue` instance per MCU in `self.gcrqs` (reuse existing queue, no duplicates).

## Runtime and Naming Contracts
- Treat names passed to `lookup_object(...)` as fixed contracts.
- Validate `lookup_object(...)` identifiers against Klipper/Happy-Hare source definitions before changing related names.
- Preserve public config-facing names unless a compatibility-safe migration is explicitly required.

## Sync and Motion Behavior
- Preserve sync modes: `gear`, `extruder`, `gear+extruder`, `extruder+gear`.
- In `gear+extruder` and `extruder+gear`, BLDC and extruder must move concurrently (not sequentially).

## Multi-MMU Behavior
- Support per-unit BLDC sections (for example `[mmu_gear_bldc unit1]`, `[mmu_gear_bldc unit2]`) with correct gate-range routing.

## Acceptance Configuration
- Ensure this BLDC config works:
```ini
[mmu_gear_bldc]
dir_pin: mmu:YAMMU_ESPOOLER_DIR_0
pwm_pin: mmu:YAMMU_ESPOOLER_PWM_0
pwm_min: 0.85
pwm_max: 1.00
hardware_pwm: True
cycle_time: 0.00005
mm_per_rev: 1.0
```
