---
name: python-refactor
description: 'Refactor Python code without behavior change. Use when asked to refactor, clean up, simplify, or reduce Python code in this project. Applies DRY, KISS, SRP — preserves all public contracts, side effects, logging, and Klipper dynamic-dispatch members.'
user-invocable: false
---

# Python Refactor

## When to Use
- User says: refactor, clean up, simplify, slim down, reduce, tidy Python code.
- File touched is `*.py` in `extras/` or `components/`.

## Priority
Correctness → readability → DRY → size.

## Procedure

1. **Read the file** in full before touching anything.
2. **Identify candidates:**
   - Unused imports, locals, private helpers with no callers.
   - Redundant assignments, temp vars that inline cleanly.
   - Duplicated logic that a shared helper would clarify (not obscure).
   - Nested guards that can flatten without hurting clarity.
3. **Apply changes:**
   - Remove dead code where clearly safe.
   - Inline temp vars when one-line result stays readable.
   - Flatten `if not X: return` guard nesting.
   - DRY only when the abstraction is simpler than the duplication.
4. **Verify:** run `python -m py_compile <file>` — must exit 0.
5. Output refactored code, then confirm compile passed.

## Hard Constraints

- No public API changes: signatures, return values, side effects, exceptions, logging output.
- No code golf or one-liners that obscure intent.
- Preserve all members accessible via dynamic dispatch:
  Klipper uses `lookup_object`, `register_event_handler`, config-facing names, and runtime string accessors.
  Remove nothing that could be reached at runtime through these paths.
- Preserve type hints (remove only when clearly safe and consistent with file style).

## Reference

Full directive list: [refactor.md](../../prompts/refactor.md)
