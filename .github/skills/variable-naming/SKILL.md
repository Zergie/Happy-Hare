---
name: variable-naming
description: 'Use when naming or renaming variables in Happy-Hare Python code. Covers naming rules, scope-aware conventions, BLDC/MMU domain terminology, and safe consistency checks before applying minimal edits.'
argument-hint: 'Describe what you are naming (new var, rename, or cleanup), language/file scope, and any style constraints.'
user-invocable: true
---

# Variable Naming

## When To Use
- Introducing new variables in Happy-Hare code.
- Renaming unclear or inconsistent variable names.
- Aligning variable names with MMU/BLDC domain terminology.
- Reviewing naming quality during refactors with minimal diffs.

## Outcome
Produce names that are clear, stable, and consistent with existing Happy-Hare style while minimizing behavior risk.

## Procedure
1. Identify scope and lifetime.
- Local temporary value: short but meaningful (`rpm`, `dt`, `gate`).
- Instance state: `self.` names that describe persistent intent (`self.bldc_mm_per_rev`).
- Constant/class-level value: uppercase snake case (`SENSOR_EXTRUDER_ENTRY`).

2. Match existing project conventions first.
- Follow file-local patterns before introducing new style.
- Prefer snake_case for variables and attributes.
- Keep domain prefixes when useful (`bldc_`, `mmu_`, `gate_`, `toolhead_`).

3. Encode meaning, not implementation trivia.
- Prefer purpose-based names (`target_rpm`, `sync_state`) over generic names (`value1`, `tmp2`).
- Use units in names when ambiguity exists (`distance_mm`, `speed_mm_s`).
- Keep boolean names readable (`is_enabled`, `has_encoder`, `should_sync`).

4. Respect runtime context and ownership.
- Keep names aligned with config keys and GCode concepts where applicable.
- For per-unit/per-gate data, include structure hints (`unit_index`, `gate_id`, `unit_to_gates`).
- Treat names passed to `lookup_object(...)` as fixed contracts.
- Resolve valid object names from Klipper or Happy-Hare source definitions before editing; do not invent or rename lookup identifiers.

5. Choose rename strategy.
- Single local variable: direct local rename.
- Shared symbol across a file/module: semantic rename with usage checks.
- Public or persisted names: preserve compatibility unless explicitly approved.

6. Apply minimal edits.
- Rename only what is needed for clarity.
- Avoid broad unrelated cleanup in the same change.
- Preserve APIs and config-facing names unless required.

7. Validate after edits.
- No broken references/imports.
- No semantic changes to control flow.
- Names remain consistent across read/write paths and logs.

## Decision Points
- Keep vs rename:
- Keep existing name if already consistent and widely referenced.
- Rename when it materially improves clarity or prevents misuse.

- lookup_object identifiers:
- If a string is used in `lookup_object(...)`, keep it unless the underlying provider name is intentionally changed in its defining source.
- Verify provider names in Klipper/Happy-Hare source before changing any related variable names or lookup strings.

- Domain specificity:
- Use domain terms when behavior is MMU-specific.
- Use generic names only for generic utility values.

- Compatibility impact:
- If name appears in saved variables, config schema, or external macro contracts, prefer backward-compatible handling.

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
