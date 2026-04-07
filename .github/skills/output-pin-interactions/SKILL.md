---
name: output-pin-interactions
description: 'Use when interacting with Klipper output pins in Happy Hare (PWM/digital), including setup_pin timing, hardware_pwm/cycle_time config, GCodeRequestQueue scheduling, and debugging pin timing errors.'
argument-hint: 'Describe the pin use case (e.g., BLDC, espooler, servo-like PWM) and the error or behavior to fix.'
user-invocable: true
---

# Output Pin Interactions

## Scope
- Workspace skill for the Happy-Hare repository.

## When To Use
- Adding or refactoring motor/pin control that uses Klipper output pins.
- Migrating from `SET_PIN` gcode calls to direct pin objects.
- Wiring BLDC or espooler-like PWM + direction control.
- Fixing timing/runtime errors around `set_pwm` / `set_digital`.
- Enabling config-driven PWM behavior (`hardware_pwm`, `cycle_time`).

## Goals
- Configure pins from module config, not ad-hoc global sections when avoidable.
- Initialize pins at the correct lifecycle stage.
- Schedule pin writes through request queues for safe MCU timing.
- Keep behavior deterministic and easy to debug.

## Procedure
1. Identify pin role and control model.
- PWM-only, digital-only, or paired PWM+DIR.
- Whether value changes are frequent or bursty.

2. Parse config from the owning section.
- Read required pin names from your feature section (for example `[mmu_gear_bldc]`).
- Parse and validate PWM options in that same section: `hardware_pwm`, `cycle_time`, min/max, defaults.

3. Create pin objects during config/init phase.
- Use `ppins = printer.lookup_object('pins')`.
- Use `ppins.setup_pin('pwm', <pin>)` or `ppins.setup_pin('digital_out', <pin>)`.
- For PWM, call `setup_cycle_time(cycle_time, hardware_pwm)`.
- Set max duration/start state (`setup_max_duration`, `setup_start_value`).
- Do not defer `setup_pin` creation to runtime handlers like `handle_connect`.

4. Create request queues per MCU.
- Prefer `output_pin.GCodeRequestQueue` when available.
- Fallback specifically to `GCodeRequestQueue` from `extras/mmu_espooler.py` when `output_pin.GCodeRequestQueue` is unavailable.
- Name the queue map `self.gcrqs`.
- Allow exactly one `GCodeRequestQueue` instance per MCU in `self.gcrqs` (reuse, do not duplicate).

5. Route all pin writes through queue callbacks.
- Public control methods should call `send_async_request(value)`.
- Queue callback receives `(print_time, value)` and performs `set_pwm` / `set_digital`.
- Enforce strict rule: no direct `set_pwm` / `set_digital` calls outside queue callbacks.

6. Handle direction changes safely.
- Optionally force PWM to zero before direction flip.
- Apply short deadtime before re-enabling PWM.
- Update cached direction state to avoid redundant writes.

7. Add guardrails and stop path.
- Clamp PWM to configured range.
- Ensure stop path always drives PWM to zero.
- Keep callbacks idempotent where practical (skip duplicate values).

8. Validate with focused checks.
- No `setup_pin` calls in runtime-only phases.
- No direct `set_pwm` / `set_digital` calls anywhere outside queue callback functions.
- Config values (`hardware_pwm`, `cycle_time`) are actually consumed.
- Emergency/disable paths can always stop the motor.

## Decision Points
- Queue source:
- If `output_pin.GCodeRequestQueue` exists, use it.
- Else use `GCodeRequestQueue` from `extras/mmu_espooler.py`.
- Queue instance policy:
- Store queues in `self.gcrqs`.
- Reuse the existing queue for that MCU and never create more than one queue per MCU.

- Configuration source:
- If feature section owns the hardware, keep all pin/PWM options there.
- Use external pin sections only when the feature must share ownership.

- Write strategy:
- High-frequency or safety-critical updates should always use queued writes.
- One-shot setup values can be configured at start-value stage.

## Completion Checklist
- [ ] Pins are configured from the feature section.
- [ ] `hardware_pwm` and `cycle_time` are parsed and applied.
- [ ] Pin objects are created during config/init, not late runtime.
- [ ] Write path uses request queue callbacks.
- [ ] Fallback queue source is `extras/mmu_espooler.py`.
- [ ] Queue map is named `self.gcrqs`.
- [ ] Exactly one queue exists per MCU (no duplicate `GCodeRequestQueue` instances).
- [ ] Stop/disable path is reliable and tested.
- [ ] No direct `set_pwm`/`set_digital` calls exist outside queue callbacks.

## References
- BLDC and queue usage pattern in `extras/mmu/mmu.py`.
- Espooler queue/callback pattern in `extras/mmu_espooler.py`.
