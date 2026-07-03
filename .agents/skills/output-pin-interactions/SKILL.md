---
name: output-pin-interactions
description: 'Use for Klipper output-pin work in Happy Hare (PWM/digital): setup_pin timing, hardware_pwm/cycle_time config, queued pin scheduling, pin timing error debug.'
argument-hint: 'Describe pin use case (BLDC, espooler, servo-like PWM) and error/behavior to fix.'
user-invocable: true
---

# Output Pin Interactions

## Scope
- Workspace skill for `Happy-Hare` repo.

## When To Use
- Adding or refactoring motor/pin control that uses Klipper output pins.
- Migrating from `SET_PIN` gcode calls to direct pin objects.
- Wiring BLDC or espooler-like PWM + direction control.
- Fixing timing/runtime errors around `set_pwm` / `set_digital`.
- Enabling config-driven PWM behavior (`hardware_pwm`, `cycle_time`).

## Goals
- Configure pins from module config, not ad-hoc global sections when avoidable.
- Initialize pins at the correct lifecycle stage.
- Schedule pin writes through proper queuing mechanisms for safe MCU timing.
- Keep behavior deterministic and easy to debug.

## Procedure
1. Identify pin role and control model.
- PWM-only, digital-only, or paired PWM+DIR.
- Whether value changes are frequent or bursty.

2. Parse config from the owning section.
- Read required pin names from feature section (example `[mmu_gear_bldc]`).
- Parse and validate PWM options in same section: `hardware_pwm`, `cycle_time`, min/max, defaults.

3. Create pin objects during config/init phase.
- Use `ppins = printer.lookup_object('pins')`.
- Use `ppins.setup_pin('pwm', <pin>)` or `ppins.setup_pin('digital_out', <pin>)`.
- For PWM, call `setup_cycle_time(cycle_time, hardware_pwm)`.
- Set max duration/start state (`setup_max_duration`, `setup_start_value`).
- Do not defer `setup_pin` creation to runtime handlers like `handle_connect`.

4. Setup queuing mechanism per MCU.
- Get `motion_queuing = printer.load_object(config, 'motion_queuing')`.
- Allocate a `syncemitter` via `motion_queuing.allocate_syncemitter(mcu, name, alloc_stepcompress=False)`.
- Store in a map keyed by MCU (reuse existing syncemitter for that MCU, do not duplicate).
- Use `syncemitter_queue_msg` from chelper to queue PWM/digital updates.

5. Route all pin writes through syncemitter queuing.
- Call `set_pwm(print_time, value)` / `set_digital(print_time, value)` to queue updates.
- Internally use `syncemitter_queue_msg` to schedule message delivery at the correct MCU time.
- Notify motion_queuing via `motion_queuing.note_mcu_movequeue_activity(print_time, is_step_gen=False)` so toolhead flushes timing.

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
- Pin writes use proper timing (print_time) and syncemitter queuing.
- Motion_queuing is notified for activity on every pin state change.
- Config values (`hardware_pwm`, `cycle_time`) are actually consumed.
- Emergency/disable paths can always stop the motor.

## Decision Points
- Queuing mechanism:
- Use `motion_queuing.allocate_syncemitter` with `syncemitter_queue_msg` for high-precision timing.
- One syncemitter per MCU; reuse existing syncemitter for that MCU if already allocated.

- Configuration source:
- If feature section owns hardware, keep all pin/PWM options there.
- Use external pin sections only when the feature must share ownership.

- Write strategy:
- High-frequency or safety-critical updates should always use queued writes via syncemitter.
- One-shot setup values can be configured at start-value stage.

## Completion Checklist
- [ ] Pins configured from feature section.
- [ ] `hardware_pwm` and `cycle_time` are parsed and applied.
- [ ] Pin objects are created during config/init, not late runtime.
- [ ] Motion queuing (syncemitter) is allocated per MCU.
- [ ] Exactly one syncemitter exists per MCU (no duplicate instances).
- [ ] Pin writes use `set_pwm`/`set_digital` with proper timing coordination.
- [ ] Motion_queuing is notified of pin activity for proper flushing.
- [ ] Stop/disable path is reliable and tested.

## References
- Klipper PWM implementation: `klippy/extras/pwm_tool.py`.
- BLDC motor control in `extras/mmu/mmu_gear_bldc.py`.
