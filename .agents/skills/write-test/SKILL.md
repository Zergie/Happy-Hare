---
name: write-test
description: "Write new Happy-Hare BLDC pytest tests. Use when adding or fixing tests for mmu.py MMU_CHECK_GATE and mmu_gear_bldc.py behavior, including fake dependency wiring, fixtures, and deterministic assertions. Triggers: write BLDC tests, add pytest for mmu gear, test MMU_CHECK_GATE, expand BLDC coverage."
argument-hint: "Describe code path, behavior under test, and whether to use command-level mmu.py harness or mmu_gear_bldc fixture harness."
user-invocable: true
---

# Write Happy-Hare BLDC Tests

## When To Use
- Need new pytest coverage for BLDC behavior in Happy-Hare.
- Need regression tests for `MMU_CHECK_GATE` command behavior in `extras/mmu/mmu.py`.
- Need focused unit coverage for `extras/mmu/mmu_gear_bldc.py` motion, sync, tachometer, or mapping behavior.
- Need to add tests that run without real firmware/hardware dependencies.

## Outcome
Produce deterministic, readable tests that follow existing project patterns and catch BLDC behavior regressions with minimal mocking noise.

## Reference Patterns
- Command-level harness pattern: `test/test_mmu_check_gate_bldc.py`
- BLDC module fixture pattern: `test/test_mmu_gear_bldc_pytest.py`
- Shared fake runtime objects: `test/support/bldc_fakes.py`

## Procedure
1. Pick test scope and harness.
- If validating command orchestration in `mmu.py` (gate state, error routing, loaded/unloaded transitions), use command-level harness pattern.
- If validating BLDC internals in `mmu_gear_bldc.py` (motion descriptors, queue/timer behavior, sync hooks, tachometer, mapping), use fixture pattern with shared fakes.

2. Create or update test file.
- Prefer existing files when behavior is closely related.
- Create a new `test/test_*.py` only when behavior is distinct and would bloat current files.
- Keep file-local helpers private (`_Name`) when only used by one file.

3. Build test scaffolding.
- Command-level harness (`mmu.py`):
  - Add minimal fake objects (`_FakeGcmd`, `_FakeMmuForCheckGate`) and call recorder methods.
  - Install import shims by replacing only required `extras.*` modules in `sys.modules`.
  - Load target module via `importlib.util.spec_from_file_location(...)`.
- Fixture harness (`mmu_gear_bldc.py`):
  - Reuse `FakeConfig`, `FakePrinter`, `FakeMmu`, `FakeToolhead`, `FakeExtruder`, `FakeMove` from `test/support/bldc_fakes.py`.
  - Use `pytest.fixture()` to build reusable `runtime_env` and BLDC unit-under-test fixtures.

4. Write scenario-focused tests.
- Name tests as `test_<unit>_<scenario>_<expected_result>`.
- Keep one primary behavior per test.
- Add a short why-docstring to every test describing why the behavior matters (regression risk or contract being protected).
- For each test, include assertion(s) on externally observable outcomes:
  - state updates (`gate_status`, `filament_pos`, `motion_state`)
  - calls/side-effects (recorded calls, monitor activation, queued descriptors)
  - error handling path (`handle_mmu_error` vs non-print path behavior)

5. Cover decision branches explicitly.
- Add branch-pair coverage: success path + failure/alternate path for each critical behavior.
- For command-level logic, include print-context branching (`is_in_print`/`is_printing`) when relevant.
- For BLDC logic, include timing-sensitive branches (kick transition, timer callback, delayed callback, stop/brake behavior).

6. Keep tests deterministic.
- Avoid real sleeps and wall-clock dependencies.
- Drive fake time through reactor methods (`monotonic`, `pause`) when timing matters.
- Avoid real hardware access or real Klipper process dependencies.

7. Run targeted tests.
- Run only impacted test files first, then broader set if needed.
- Example:
  - `pytest test/test_mmu_check_gate_bldc.py -q`
  - `pytest test/test_mmu_gear_bldc_pytest.py -q`

8. Validate quality gates before finishing.
- Ensure no flaky timing assumptions.
- Ensure assertions verify behavior, not implementation trivia.
- Ensure helper fakes are minimal and scoped.

## Decision Points
- Harness choice:
  - Use command-level harness for command orchestration contracts.
  - Use fixture harness for BLDC internal mechanics and component interactions.

- New test file vs existing file:
  - Add to existing file if behavior shares same fake setup and domain.
  - Split to new file if setup diverges significantly or readability drops.

- Mock style:
  - Prefer project fakes over deep mocking.
  - Use `MagicMock` selectively for ownership and callback interaction checks.

## Completion Checklist
- [ ] Correct harness selected for behavior under test.
- [ ] Test names reflect scenario and expected outcome.
- [ ] Every test includes a concise why-docstring.
- [ ] Branch-pair coverage exists for each critical behavior (success + failure/alternate).
- [ ] Assertions verify observable state/calls/errors.
- [ ] Timing-related tests use fake reactor time only.
- [ ] No real hardware/process dependencies introduced.
- [ ] Targeted pytest run completed and passing.

## Example Prompts
- "Write tests for MMU_CHECK_GATE when BLDC load fails during print and outside print."
- "Add pytest coverage for mmu_gear_bldc brake_to_stop timing and queued descriptors."
- "Create tests for direction_map validation and gate mapping behavior in BLDC gear module."
