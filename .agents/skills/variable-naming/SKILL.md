---
name: variable-naming
description: 'Use for var naming/renaming in Happy-Hare Python. Rules: scope, BLDC/MMU terms, consistency checks, minimal edits.'
argument-hint: 'Describe task: new var | rename | cleanup. Add file/scope + style constraints.'
user-invocable: true
---

# Variable Naming

## When To Use
- Add new vars in Happy-Hare code.
- Rename unclear/inconsistent vars.
- Align names with MMU/BLDC domain terms.
- Review naming during refactor with minimal diff.

## Outcome
Produce names clear + stable + style-consistent. Keep behavior risk low.

## Procedure
1. Identify scope and lifetime.
- Local temp: short + meaningful (`rpm`, `dt`, `gate`).
- Instance state: `self.` names for persistent intent (`self.bldc_mm_per_rev`).
- Const/class-level: UPPER_SNAKE_CASE (`SENSOR_EXTRUDER_ENTRY`).

2. Match existing project conventions first.
- Follow file-local patterns first.
- Prefer snake_case for variables and attributes.
- Keep domain prefixes when useful (`bldc_`, `mmu_`, `gate_`, `toolhead_`).

3. Encode meaning, not implementation trivia.
- Prefer purpose names (`target_rpm`, `sync_state`) > generic (`value1`, `tmp2`).
- Add units when ambiguous (`distance_mm`, `speed_mm_s`).
- Bool names readable (`is_enabled`, `has_encoder`, `should_sync`).

4. Respect runtime context and ownership.
- Keep names aligned with config keys + GCode concepts.
- For per-unit/per-gate data, include structure hints (`unit_index`, `gate_id`, `unit_to_gates`).
- Treat names passed to `lookup_object(...)` as fixed contracts.
- Resolve valid object names from Klipper/Happy-Hare source before edit. Never invent/rename lookup identifiers.

5. Choose rename strategy.
- Single local var: direct local rename.
- Shared symbol across file/module: semantic rename + usage checks.
- Public/persisted names: keep compatibility unless explicitly approved.

6. Apply minimal edits.
- Rename only needed symbols.
- Avoid broad unrelated cleanup same change.
- Preserve APIs and config-facing names unless required.

7. Validate after edits.
- No broken references/imports.
- No semantic changes to control flow.
- Names stay consistent across read/write paths + logs.

## Decision Points
- Keep vs rename:
- Keep existing name if consistent + widely referenced.
- Rename when clarity gain material or misuse risk drops.

- lookup_object identifiers:
- If string used in `lookup_object(...)`, keep unless provider name intentionally changed in defining source.
- Verify provider names in Klipper/Happy-Hare source before changing related var names or lookup strings.

- Domain specificity:
- Use domain terms when behavior is MMU-specific.
- Use generic names only for generic utility values.

- Compatibility impact:
- If name appears in saved vars, config schema, or external macro contracts -> prefer backward-compatible handling.

## Quality Checklist
- [ ] Name reflects intent and scope.
- [ ] Style matches local Happy-Hare conventions.
- [ ] Units/booleans are explicit when needed.
- [ ] No unnecessary refactor churn.
- [ ] All references remain valid after rename.
- [ ] External-facing names preserved or compatibility handled.
- [ ] Any `lookup_object(...)` names were validated against source definitions and treated as fixed identifiers.

## Quick Examples
- Bad: `x`, `tmp`, `foo_flag`
- Better: `target_rpm`, `poll_interval_s`, `is_sync_active`
- Good MMU domain: `gate_homing_endstop`, `bldc_mm_per_rev`, `espooler_assist_burst_power`
