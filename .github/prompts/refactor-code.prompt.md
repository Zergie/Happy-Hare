---
name: "refactor code"
description: "Refactor code to be about 15% shorter while preserving readability and behavior"
argument-hint: "Code, file path, or symbol to refactor (plus constraints)"
agent: Python Refactor
---
Refactor the provided code to be about 15% shorter without reducing readability.

Requirements:
- Preserve behavior, side effects, and public API unless I explicitly allow changes.
- Remove variables aggressively.
- Inline wherever possible, especially single-use pass-through variables.
- Prefer simpler control flow and early returns over nested branches.
- Remove dead code, redundant guards, and semantic no-op assignments.
- Do not mirror configuration values (e.g. `pwm_min`, `cycle_time`) in `get_status` return dicts. Status is runtime state, not config echo.
- Keep names clear; do not trade readability for brevity.
- Use positional arguments instead of redundant keyword arguments when the argument name matches the parameter name and the call is unambiguous. Example: prefer `self.stop(print_time)` over `self.stop(print_time=print_time)`.
- Keep shared arguments in the same or near-same position across related method signatures.
- Keep positional call ordering aligned with the signature convention for shared arguments across related methods.
- Only deviate from argument-position consistency when required for public API compatibility, override contracts, or clear readability gains.

## Argument Ordering
- Prefer consistent parameter index for shared arguments across related methods ("near-same" means one-slot drift only when required).
- If you deviate, mention the reason briefly in the refactor summary.

BAD:
```python
def move(self, gate, speed, print_time):
	...

def stop(self, print_time, gate):
	...

self.move(gate, speed, print_time)
self.stop(print_time, gate)
```

GOOD:
```python
def move(self, gate, print_time, speed):
	...

def stop(self, gate, print_time):
	...

self.move(gate, print_time, speed)
self.stop(gate, print_time)
```

## Wrapper Methods
- Remove private wrapper methods that only call `getattr`, forward one attribute/method access, or rename another object member.
- Inline them at the call site.
- Do not keep wrappers solely for "safety" or "readability".
- If the wrapped member is required and expected to exist, access it directly.
- Only keep a wrapper if it adds validation, fallback logic, normalization, or is used by external/dynamic dispatch.

## Unused Optional Parameters
- Remove optional parameters (parameters with default values) that are never passed by any caller.
- Remove all corresponding internal uses of that parameter.
- Do not remove required parameters or parameters that are part of a public/overridable API contract.

## Method Ownership
- If a method is only called by one collaborating class, move the method definition into that consuming class.
- Keep the method on its original class only if it is part of public API, dynamic dispatch surface, shared by multiple callers, or required for compatibility contracts.
- Prefer consumer-local private helpers for monitor-only/orchestrator-only logic to improve cohesion and reduce cross-class coupling.

Example:

Bad:
```python
def _get_selected_gate(self):
	return getattr(self.mmu, 'gate_selected', None)

gate = self._get_selected_gate()
```

Good:
```python
gate = self.mmu.gate_selected if gate is None else gate
```

BAD:
```python
def _estimate_systime_from_print_time(self, print_time):
	mcu = self.mcu_pwm_pin.get_mcu()
	clocksync = getattr(mcu, '_clocksync', None)
	if clocksync is None or not hasattr(clocksync, 'estimate_clock_systime'):
		return self.reactor.NOW
	reqclock = mcu.print_time_to_clock(max(0., print_time))
	return max(self.reactor.NOW, clocksync.estimate_clock_systime(reqclock))

def estimate_systime_from_print_time(self, print_time):
	return self._estimate_systime_from_print_time(print_time)
```

Good:
```python
def estimate_systime_from_print_time(self, print_time):
	mcu = self.mcu_pwm_pin.get_mcu()
	clocksync = getattr(mcu, '_clocksync', None)
	if clocksync is None or not hasattr(clocksync, 'estimate_clock_systime'):
		return self.reactor.NOW
	reqclock = mcu.print_time_to_clock(max(0., print_time))
	return max(self.reactor.NOW, clocksync.estimate_clock_systime(reqclock))
```

Working style:
1. Read full target scope first (function/file/module as provided).
2. Apply minimal-diff refactor directly.
3. Do not change unrelated code.
4. Validate with available lint/tests or compile checks when possible.
5. All pylint test cases must pass without adding `# pylint: disable` comments. If inlining a private member access across class boundaries would trigger a `protected-access` warning, keep the access via a public method or attribute instead.

Output format:
1. Summary of key simplifications.
2. List of variables removed or inlined.
3. Risks or behavior-sensitive areas checked.
4. Exact files changed.
