# Project Guidelines

## Scope
- Workspace has `klipper/`, `Happy-Hare/`, `printer_data/`, `Happy-Hare.wiki/`.
- Main dev target: `Happy-Hare/`.
- Python work: use `Happy-Hare/extras/` as main source.
- `klipper/` files are reference-only.
- Do not edit Klipper core (`klipper/klippy/**`, `klipper/src/**`) for Happy-Hare feature work.
- Use workspace-relative paths in responses, plans, and file refs (example: `klipper/`, not `c:/GIT/klipper/`).

## Code Style
- Keep changes small and local to owning module.
- Match file-local naming and structure in `Happy-Hare/extras/**/*.py`.
- Keep config-facing and integration contract names (example: `lookup_object(...)` identifiers) unless compatibility-safe migration is required.
- For naming, follow `.github/skills/variable-naming/SKILL.md`.

## Architecture
- BLDC motor logic lives in `Happy-Hare/extras/mmu/mmu_gear_bldc.py`.
- `Happy-Hare/extras/mmu/mmu.py` stays integration/orchestration-focused.
- Keep sync behavior contracts in `Happy-Hare/extras/mmu/mmu_sync_controller.py`:
  - Preserve modes `gear`, `extruder`, `gear+extruder`, `extruder+gear`.
  - In `gear+extruder` and `extruder+gear`, moves must run concurrent, not sequential.
- For BLDC constraints, follow `.github/instructions/bldc-guardrails.instructions.md`.

## Output Pin Conventions
- Use queued pin writes for BLDC pwm/digital paths.
- Do not call `set_pwm` or `set_digital` outside queue callbacks.
- Use `output_pin.GCodeRequestQueue` when available; fallback queue behavior must match project guardrails.
- Keep one queue per MCU; reuse it.
- Full details: `.github/skills/output-pin-interactions/SKILL.md`.

## Build and Test
- Preferred workflow: use existing VS Code tasks in `Happy-Hare` workspace folder:
  - `Export Code to Test Rig`
  - `Export Startup.gcode to Test Rig`
  - `Start on Test Rig`
  - `Stop on Test Rig`
  - `Run Startup.gcode on Test Rig`
  - `Read klippy.log on Test Rig`
- For test execution guidance in chat, use `.github/prompts/run-test-rig.prompt.md`.

## Docs (Link, Do Not Duplicate)
- Project overview: `README.md`
- Contribution process: `.github/CONTRIBUTING.md`
- Wiki deep dives: `Happy-Hare.wiki/` (architecture, hardware/configuration, troubleshooting)
- Config references: wiki pages for `mmu_hardware.cfg`, `mmu.cfg`, and related setup guides

## Common Pitfalls
- Breaking `lookup_object(...)` identifier contracts.
- Adding duplicate queue instances per MCU.
- Moving BLDC logic into orchestrator modules.
- Changing sync-mode behavior from concurrent to sequential in combined modes.

## Caveman Mode
Respond terse like smart caveman. All technical substance stay. Only fluff die.

Rules:
- Drop: articles (a/an/the), filler (just/really/basically), pleasantries, hedging
- Fragments OK. Short synonyms. Technical terms exact. Code unchanged.
- Pattern: [thing] [action] [reason]. [next step].
- Not: "Sure! I'd be happy to help you with that."
- Yes: "Bug in auth middleware. Fix:"

Switch level: /caveman lite|full|ultra|wenyan
Stop: "stop caveman" or "normal mode"

Auto-Clarity: drop caveman for security warnings, irreversible actions, user confused. Resume after.

Boundaries: code/commits/PRs written normal.
