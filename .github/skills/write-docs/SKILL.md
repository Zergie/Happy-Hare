---
name: write-docs
description: 'Write and maintain Happy Hare MMU operation deep-dive docs. Use when asked to document MMU command behavior, motor modes, movement flow, sync semantics, BLDC interactions, defaults, and gotchas (for example: MMU_TEST_MOVE, MMU_CHECK_GATE, load/unload operations). Produces strict docs/MMU_<COMMAND>.md files and updates Agents.md only when creating a brand-new doc.'
argument-hint: 'MMU command to document (for example: MMU_TEST_MOVE)'
user-invocable: true
ag
---

# Write Docs

## Purpose

Create accurate, source-anchored MMU operation documentation for Happy Hare.

Primary outcomes:
- start from `Happy-Hare.wiki` as the initial behavior/context reference
- explain operation behavior clearly (software path + physical expectation)
- explain every mode/variant the operation exposes
- map params/modes to concrete execution paths
- explain what to expect when running the operation
- explain how and when the operation is used
- capture defaults, constraints, and gotchas
- keep discoverability anchors in `Agents.md` only when required

## Strict Filename Rule

- **One command per file.**
- **Required path format:** `docs/MMU_<COMMAND>.md`
- `<COMMAND>` must match the command token exactly (uppercase + underscores), e.g.:
  - `MMU_TEST_MOVE` -> `docs/MMU_TEST_MOVE.md`
  - `MMU_CHECK_GATE` -> `docs/MMU_CHECK_GATE.md`
- Do not group multiple commands into one doc unless explicitly requested.

## When to Use

Use this skill when asked to:
- document an MMU command/operation
- explain operation modes/variants (not limited to motor modes)
- create or update `docs/MMU_*.md`

## Inputs

- Command name (required), e.g. `MMU_TEST_MOVE`
- Whether to create new file or update existing file. If not specified by the user, check whether `docs/MMU_<COMMAND>.md` already exists in the codebase. If it exists, treat the task as an update; if it does not, treat it as a new file creation.
- Scope to cover (modes/branches/edge cases)

## Procedure

1. **Locate command entrypoints**
   - Start with `Happy-Hare.wiki` pages relevant to the operation to capture intended user-facing behavior and purpose.
   - Find registration/handler in `extras/mmu/mmu.py`.
   - If the command handler is not found in `extras/mmu/mmu.py`, search the broader codebase (for example other files under `extras/mmu/`) before proceeding. If still not found, state this explicitly in the doc and note that source anchoring is incomplete.
   - Identify shared helpers (`_move_cmd`, `trace_filament_move`, homing variants, sync helpers).

2. **Trace execution end-to-end**
   - Explore the codebase beyond the entrypoint to build deep understanding of operation intent and side effects.
   - Follow each branch to concrete move call sites.
   - Track sync-mode transitions (`MmuToolHead`), driver/follower ownership, queue behavior.
   - If BLDC path exists, trace BLDC routing, hook/monitor behavior, and stop semantics.

3. **Establish operation purpose and usage**
   - Explain what the operation is for in normal workflows.
   - Explain when operators should use it and when they should avoid it.
   - Reconcile wiki intent with code behavior; when they differ, prefer code and call out mismatch.

4. **Extract decision points/defaults**
   - Legal params and validation.
   - Mode/path defaults (speed/accel/current/wait).
   - Context-dependent behavior (bypass, selector state, print state, homing constraints).

5. **Enumerate and explain every operation mode**
   - Identify all modes/variants the operation supports (for example: `MOTOR`, `TYPE`, `STATE`, direction/context variants).
   - For each mode: explain trigger/selection, execution path, and expected observable result.
   - If a mode is unsupported in a context (for example homing/synced constraints), state it explicitly.

6. **Write operation doc (generic operation-focused explanation)**
   - Target strict filename: `docs/MMU_<COMMAND>.md`.
   - Focus on operation behavior and source anchors.
   - Include:
     - what operation is used for
     - how to use it (key parameters and common invocation patterns)
     - what to expect physically/logically per mode
   - No required test-rig validation section.

7. **Anchor discoverability conditionally**
   - Update `Agents.md` **only when** creating a brand-new `docs/MMU_<COMMAND>.md` file.
   - If updating an existing doc, do not modify anchors unless explicitly requested.

## Branching Logic

- If operation has multiple modes (any selector/parameter): add concise mode matrix (how selected, what runs, what to expect, when to use).
- If operation has homing + non-homing paths: split and describe both.
- If BLDC path differs from stepper path: document differences explicitly.
- If behavior differs in print vs non-print: split by context.
- When wiki and code differ, prefer code behavior and call out the mismatch in the doc. Only pause to request targeted clarification when the code itself is ambiguous (for example multiple conflicting branches with no clear winner).

## Quality Gate

Before finalizing, confirm:
1. Every operation mode/branch is covered.
2. Claims are source-anchored (wiki context + function/module pointers).
3. Terminology matches code (`gear+extruder` vs internal `extruder+gear`).
4. Concurrency semantics are explicit where applicable.
5. Defaults/overrides are documented.
6. Doc explains operation usage (what it is for, how to run it, when to use it).
7. Doc states expected outcomes/observations per mode.
8. Filename follows strict rule exactly.
9. `Agents.md` edited only if doc is brand-new.

## Output Contract

- Primary: `docs/MMU_<COMMAND>.md`
- Secondary: `Agents.md` update only for brand-new docs
- Final summary: what was verified, where documented, unresolved ambiguities
