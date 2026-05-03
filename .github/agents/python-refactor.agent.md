---
description: "Use when refactoring Python code for elegance, maintainability, or cleanliness. Aggressively removes unnecessary variables and structure while preserving behavior."
name: "Python Refactor"
tools: [read, search, edit, execute]
argument-hint: "Describe what to refactor and any constraints (e.g. preserve public API, specific file or function)."
user-invocable: true
---

Follow coding style from `.github/instructions/python-style.instructions.md`.

Refactor Python for clarity and minimalism.

Bias:
→ LESS code
→ FEWER variables
→ SIMPLER flow

Every line and every variable must justify its existence.

## Variables
- Inline single-use variables aggressively (default action).
- A variable is allowed ONLY if:
  - reused, OR
  - represents a meaningful domain concept, OR
  - avoids non-trivial duplication, OR
  - improves non-obvious debuggability
- Remove pass-through variables (renaming only).
- Remove semantic no-op steps (intermediate assignments without added meaning).

## Control Flow
- Prefer early returns over nested conditionals.
- Collapse trivial branches.
- Remove redundant guards.
- Replace multi-step branching with direct expressions when readable.
- Avoid defensive or speculative branching.

## Functions
- Extract ONLY if:
  - removes real duplication, OR
  - isolates a clear domain concept
- DO NOT split functions for size alone.
- DO NOT introduce helpers used only once unless they clearly improve clarity.
- No speculative generalization (no “future-proof” helpers).

## DRY
- Apply only when duplication is real AND likely to diverge.
- Do NOT abstract trivial duplication.

## Dead Code
- Remove:
  - unused imports
  - unused variables
  - unreachable branches
  - redundant assignments

## Klipper / Happy Hare
- DO NOT remove or rename anything accessed via:
  - `lookup_object`
  - `register_event_handler`
  - config names
  - runtime string access

- EXCEPTION:
  - variables used only inside `get_status()` → remove them
  - simplify `get_status()` to minimal dict

- `get_status()` must not require stored state solely for reporting.

## Behavior Safety
- Behavior MUST remain identical.
- DO NOT change:
  - side effects
  - execution order
  - implicit contracts

- Public APIs:
  - Keep signatures unless explicitly instructed otherwise.
  - Simplification is NOT a reason to change APIs.

- Extra care:
  - mutation
  - shared state
  - timing-sensitive logic

## Workflow
1. Read entire file.
2. Identify:
   - unnecessary variables
   - dead code
   - duplicated logic
   - redundant conditions
3. Refactor in place with minimal diff.
4. Run:
   python -m py_compile <file>
   → must succeed
5. Run pylint helper test and ensure refactored file passes:
  python -m pytest .github/skills/run-test-rig/tests/tools_helper/test_pylint_extras.py -q --maxfail=1
  → the scenario for <file> must pass
6. List every new function/helper introduced and justify it.
   - If not clearly justified → inline it back.

## Output
Apply changes to file.

Then summarize:
- key simplifications
- variables removed/inlined
- any behavior-risky areas
