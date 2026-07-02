---
description: "Use when editing test/test_mmu_gcode.py and related MMU gcode harness tests. Prefer config-driven initialization and mocked config.printer wiring over monkeypatching Mmu instance/class methods."
applyTo: "test/test_mmu_gcode.py"
---

# MMU Test Harness Method Override Policy

Use this policy for MMU harness tests in `test/test_mmu_gcode.py`.

## Preferred approach

- Do not overwrite methods on `Mmu` class or `mmu` instances when building test setup.
- Prefer deterministic behavior through mocked `config` values and mocked `config.printer` objects.
- It is acceptable to use mocked `config.printer` object graphs to satisfy dependency wiring.

## If override cannot be avoided

- Keep override as a slim traceability layer only.
- Wrapper must call original method exactly once.
- Forward arguments unmodified (`*args`, `**kwargs`) with no transformation.
- Preserve return value from original method unchanged.
- Do not add behavior changes beyond minimal tracing (for example, append call metadata).

## Review checks

- Ask: can this be solved by `config`/`config.printer` setup first?
- Reject override if it changes control flow, argument semantics, side effects, or return semantics.
- Prefer per-test local wrappers over broad fixture- or module-wide monkeypatching.
