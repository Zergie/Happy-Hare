# Project Guidelines

## Scope
- This workspace includes `klipper/`, `Happy-Hare/`, `printer_data/`, and `Happy-Hare.wiki/`.
- Primary development target is `Happy-Hare/`.
- For Python implementation work, treat `Happy-Hare/extras/` as the primary source location.
- Files under `klipper/` are reference-only.
- Do not edit Klipper core (`klipper/klippy/**`, `klipper/src/**`) for Happy-Hare feature work.
- Use workspace-relative paths in all responses, plans, and file references (for example `klipper/` instead of `c:/GIT/klipper/`).

## Code Style
- Keep changes minimal and local to the module that owns behavior.
- Match existing file-local naming and structure in `Happy-Hare/extras/**/*.py`.
- Preserve config-facing and integration contract names (for example `lookup_object(...)` identifiers) unless a compatibility-safe migration is explicitly required.
- For naming decisions, follow `.github/skills/variable-naming/SKILL.md`.

## Architecture
- BLDC motor logic belongs in `Happy-Hare/extras/mmu/mmu_gear_bldc.py`.
- `Happy-Hare/extras/mmu/mmu.py` should remain integration/orchestration focused.
- Keep sync behavior contracts in `Happy-Hare/extras/mmu/mmu_sync_controller.py`:
  - Preserve modes `gear`, `extruder`, `gear+extruder`, `extruder+gear`.
  - In `gear+extruder` and `extruder+gear`, moves must be concurrent, not sequential.
- For BLDC implementation constraints, follow `.github/instructions/bldc-guardrails.instructions.md`.

## Output Pin Conventions
- Use queued pin writes for BLDC pwm/digital paths.
- Do not call `set_pwm` or `set_digital` outside queue callbacks.
- Use `output_pin.GCodeRequestQueue` when available; fallback queue behavior must match project guardrails.
- Keep exactly one queue per MCU and reuse it.
- For full details, see `.github/skills/output-pin-interactions/SKILL.md`.

## Build and Test
- Preferred workflow uses existing VS Code tasks in the `Happy-Hare` workspace folder:
  - `Export Code to Test Rig`
  - `Export Startup.gcode to Test Rig`
  - `Start on Test Rig`
  - `Stop on Test Rig`
  - `Run Startup.gcode on Test Rig`
  - `Read klippy.log on Test Rig`
- For test execution guidance in chat, use `.github/prompts/run-test-rig.prompt.md`.

## Docs (Link, Do Not Duplicate)
- Main project overview: `README.md`
- Contribution process: `.github/CONTRIBUTING.md`
- Wiki deep dives: `Happy-Hare.wiki/` (architecture, hardware/configuration, troubleshooting)
- Config references: wiki pages for `mmu_hardware.cfg`, `mmu.cfg`, and related setup guides

## Common Pitfalls
- Breaking `lookup_object(...)` identifier contracts.
- Adding duplicate queue instances per MCU.
- Moving BLDC logic into orchestrator modules.
- Changing sync-mode behavior from concurrent to sequential in combined modes.
