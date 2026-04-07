## Plan: BLDC Replacement For stepper_mmu_gear With Sync Support

Implement BLDC gear-drive support directly in MMU runtime so stepper_mmu_gear behavior is replaced by DIR+PWM+tach control, while preserving Happy Hare extruder-sync workflows. No enable flag is used; BLDC mode is selected by presence of BLDC gear config.

Primary requirement: preserve existing Happy Hare motion semantics.
Secondary requirement: BLDC integration must honor commanded distance, speed, and acceleration behavior used throughout MMU logic.
Runtime note: Happy-Hare is an extra module symlinked from Happy-Hare/extras into klipper/klippy/extras at runtime. Keep implementation in Happy-Hare paths, but import Klipper modules with this runtime context in mind.

**Steps**
1. Phase 1 - BLDC configuration model (no enable flag)
1. Add required BLDC gear config parsing in Happy-Hare/extras/mmu/mmu.py using a dedicated section (recommended: [mmu_gear_bldc]). Required keys: dir pin object name, pwm pin object name, tach pin, tach_ppr default 9, max_rpm default 6000, pwm min/max.
2. Select BLDC behavior by config presence only; if BLDC section exists, gear motor path runs BLDC logic. If absent, retain current stepper path unchanged.
3. Enforce mutual exclusivity for gear drive definitions: if both [stepper_mmu_gear] and [mmu_gear_bldc] are defined, fail startup with a clear configuration error and remediation message.
4. Remove hard dependency on [stepper_mmu_gear] when BLDC config is present: startup must succeed with BLDC-only gear configuration.
5. Patch Happy-Hare/extras/mmu_machine.py gear-rail initialization so gear section resolution supports either:
	- stepper gear rail from [stepper_mmu_gear] (legacy), or
	- BLDC gear actuator path from [mmu_gear_bldc] without loading a stepper rail section.
6. For BLDC-only mode, provide a gear motion compatibility object/path that satisfies MMU movement/state APIs without requiring stepper pin/driver definitions.
2. Phase 2 - Runtime BLDC controller in MMU
1. Add BLDC control logic in Happy-Hare/extras/mmu/mmu_gear_bldc.py and keep Happy-Hare/extras/mmu/mmu.py as integration/orchestration only.
2. Reuse pulse counter frequency pattern from klipper/klippy/extras/fan.py and klipper/klippy/extras/pulse_counter.py, adapting conversion for 9 PPR.
3. For BLDC PWM and direction pins, follow the espooler_control_macro/pin-control pattern in Happy-Hare/extras/mmu/mmu.py and Happy-Hare/extras/mmu_espooler.py, using proper pwm/digital_out pin handling and queued updates.
4. Implement a motion-profile adapter that converts requested distance/speed/accel into a BLDC trajectory (trapezoid/triangle equivalent):
	- Respect requested max speed and accel limits for every move.
	- Integrate target speed over time to hit commanded distance within tolerance.
	- Track and expose residual error if interrupted (for deterministic recovery).
5. Add mm_per_rev support with two paths:
	- Calibrated path: new calibration routine/command to determine mm_per_rev from measured filament movement.
	- Manual path: configuration parameter allowing users to provide mm_per_rev directly if known from mechanical design.
6. Persist calibrated mm_per_rev and expose reset/recalibrate workflow.
3. Phase 3 - Replace gear-drive movement path
1. In trace_filament_move(), for motor modes that include gear (gear, gear+extruder), map requested dist/speed into BLDC direction, PWM duty, and run duration.
2. Preserve existing movement accounting (actual distance, encoder measurement delta, logging) so higher-level MMU behavior remains stable.
3. Keep speed_override, gate-specific speed factors, and accel wrappers behaviorally equivalent to the stepper path so user tuning in mmu_parameters.cfg remains meaningful.
4. Homing in BLDC mode must be sensor-based only. Non-sensor or stepper-touch style homing paths must be rejected with explicit user-facing configuration/runtime errors.
	- Allowed sensor-based homing: physical sensor endstops such as gate sensors, extruder entry, toolhead, and compression when configured.
	- Disallowed for BLDC homing: none, collision, and touch/stallguard-style paths (including gear_touch/ext_touch style virtual touch paths).
4. Phase 4 - Extruder sync support with BLDC
1. Implement sync as an event-driven BLDC follower loop, not trapq ownership of a physical gear stepper. Reuse sync lifecycle events (mmu:synced, mmu:unsynced) already emitted by Happy-Hare/extras/mmu_machine.py and already consumed by Happy-Hare/extras/mmu/mmu_sync_feedback_manager.py.
2. On mmu:synced: initialize BLDC sync state, reset PI integrator, then register an extruder movement callback (thresholded) to drive follower updates.
3. On mmu:unsynced: unregister extruder callback and force immediate BLDC stop (PWM=0), then clear sync state.
4. In each extruder movement callback: compute extruder speed v_ext = delta_ext_mm / dt, map to target rpm as rpm_target = 60 * abs(v_ext) / mm_per_rev, clamp to max_rpm=6000, set DIR from sign(v_ext), then command PWM using feed-forward only in v1.
5. For v1, use feed-forward PWM plus tach monitoring only. Do not implement any tach correction loop in v1.
6. Use tach conversion consistent with Klipper fan tach handling in klipper/klippy/extras/fan.py: rpm = freq * 30 / ppr (with ppr=9).
7. Keep Happy-Hare/extras/mmu_machine.py changes minimal: preserve sync interfaces/states, only bypass or guard assumptions that a real gear stepper must provide torque while in GEAR_SYNCED_TO_EXTRUDER.
8. Add synced-mode safety constraints: direction-change deadtime, rpm clamp, overspeed cutoff, and fail-safe stop when sync confidence is lost.
9. In synced mode, enforce extruder-relative motion semantics: BLDC target profile must be derived from extruder motion delta and timing so effective filament transport matches expected synchronized distance and acceleration envelopes.
5. Phase 5 - Guard stepper-only controls
1. Guard TMC current logic and stepper_enable paths in Happy-Hare/extras/mmu/mmu.py so BLDC gear mode does not call SET_TMC_CURRENT for gear or depend on stepper enable state.
2. Keep MMU_MOTORS_OFF semantics strict in BLDC mode: always force PWM=0 and neutral safe direction state.
6. Phase 6 - Runtime config updates
1. Add BLDC section and pin wiring entries under printer_data/base/mmu_hardware.cfg.
2. Add BLDC tuning parameters (pwm shaping, sync caps) under printer_data/base/mmu_parameters.cfg.
3. Support multi-mmu BLDC configurations where more than one MMU unit is active at once. For example:
	- [mmu_machine] num_gates: 4,4
	- [mmu_gear_bldc unit1] controls first unit gates (0-3)
	- [mmu_gear_bldc unit2] controls second unit gates (4-7)

**Relevant files**
- Happy-Hare/extras/mmu/mmu.py — BLDC integration/orchestration, gear move mapping hooks, synced-move integration, stepper-only guards.
- Happy-Hare/extras/mmu/mmu_gear_bldc.py — BLDC control implementation (PWM/DIR/tach, motion profile, sync-follow control and safety handling).
- Happy-Hare/extras/mmu_machine.py — remove mandatory stepper_mmu_gear dependency in BLDC mode, provide compatible gear motion path, and keep sync state interactions stable.
- printer_data/base/mmu_hardware.cfg — BLDC gear section and pin object wiring.
- printer_data/base/mmu_parameters.cfg — BLDC runtime tuning defaults and safety thresholds.
- klipper/klippy/extras/fan.py — tachometer conversion reference pattern.
- klipper/klippy/extras/pulse_counter.py — frequency counter API reference.
- Happy-Hare/extras/mmu_espooler.py — GCodeRequestQueue and pwm/digital_out pin scheduling reference.

**Verification**
1. Restart Klipper with BLDC-only gear config (no [stepper_mmu_gear]) and confirm startup config parses cleanly.
2. Run MMU_MOTORS_OFF and verify immediate PWM=0 and stopped BLDC.
3. Run forward/reverse gear moves and verify DIR pin polarity correctness (CW/CCW).
4. Run commanded speeds across range and verify PWM scaling and cap behavior.
5. Run acceleration step tests (low/high accel) and verify measured ramp rates respect configured accel limits.
6. Run fixed-distance moves at multiple speed/accel combinations and verify distance error stays within defined tolerance.
7. Verify tach RPM is non-zero and plausible for 9 PPR and below configured max_rpm=6000.
8. Validate sync lifecycle behavior: confirm mmu:synced starts BLDC follower updates and mmu:unsynced always stops BLDC and unregisters callbacks.
9. Validate synchronized printing path: execute motor="synced" style operation and confirm extruder motion remains coordinated with BLDC assist and no sync fault.
10. Simulate tach failure during synced movement and verify fail-safe stop and clear diagnostics.
11. Simulate abrupt direction reversals and verify deadtime handling prevents unsafe reverse switching.
12. Run load/unload/toolchange smoke tests to confirm no regression in selector/extruder workflows.
13. Confirm legacy stepper configuration still starts and operates unchanged when [stepper_mmu_gear] is used.
14. Confirm startup hard-fails with clear error when both [stepper_mmu_gear] and [mmu_gear_bldc] are present.
15. Validate mm_per_rev calibration routine, persistence, manual override behavior, and reset workflow.
16. Validate BLDC homing succeeds only with configured supported sensors and fails clearly for unsupported/non-sensor homing modes.
17. Validate multi-mmu BLDC routing with num_gates: 4,4 so unit1 operations affect only gates 0-3 and unit2 operations affect only gates 4-7.

**Decisions**
- No enabled flag: BLDC behavior is selected by BLDC config presence.
- If both [stepper_mmu_gear] and [mmu_gear_bldc] are present, startup must error (mutually exclusive configuration).
- BLDC must support Happy Hare extruder sync behavior; synced path is in scope for v1.
- Scope is Happy-Hare extras code (symlinked into Klipper extras at runtime), not direct edits to Klipper source files for BLDC feature logic.
- Tach remains monitoring/safety in v1; all tach correction loops are deferred to v2.
- Distance/speed/acceleration semantics are first-class compatibility constraints and cannot be approximated away.
- BLDC mode must not require the [stepper_mmu_gear] section.
- BLDC mm_per_rev must be obtainable by calibration and overridable by direct config value.
- BLDC homing must be sensor-based.
- BLDC mode must support multi-mmu operation with per-unit [mmu_gear_bldc unitN] sections mapped to the corresponding gate ranges.

**Further Considerations**
1. V2: add tach correction loops (PI or equivalent closed-loop sync correction), optionally augmented by filament encoder feedback.
2. If trapq-linked assumptions in mmu_machine.py remain brittle, a dedicated BLDC gear abstraction may be required in a follow-up refactor.
