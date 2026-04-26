## Plan: Add Min-RPM Plused Fallback

Introduce a BLDC plused control path that auto-activates when tach control targets speed below controllable minimum RPM, while preserving existing continuous tach behavior for normal speeds. Implement only in BLDC ownership layer, keep queued pin-write guarantees, and add short config keys for threshold/timing tuning.

**Steps**
1. Phase 1 - Baseline and scope lock
2. Confirm current low-speed control flow in Happy-Hare/extras/mmu/mmu_gear_bldc.py: locate `CONTROL_MIN_RPM` (expect 150 RPM), trace _is_control_eligible(), _set_rpm(), _handle_tachometer(), _set_pin(), and rpm-to-pwm mapping path. This phase blocks all later phases.
3. Define explicit fallback contract: if tach control enabled and target RPM < (pulse_rpm - 100 RPM hysteresis), switch to plused output strategy instead of disabling control; if RPM >= pulse_rpm, exit to continuous control (depends on step 2).
4. Freeze non-goals: no sync-mode semantic changes, no mmu.py orchestration changes, no lookup_object identifier changes (depends on step 2).

5. Phase 2 - Config model with short names
6. Add short config keys in Happy-Hare/extras/mmu/mmu_gear_bldc.py parser and defaults:
7. pulse_rpm: fallback threshold RPM for plused mode activation.
8. pulse_ms: base pulse period in milliseconds.
9. Optional only if needed by implementation after prototyping: pulse_on_min (minimum ON time safety floor), pulse_pwm (fixed pulse amplitude). Keep keys short and avoid adding unused knobs.
10. Keep existing keys fully backward compatible and preserve current defaults when pulse fallback not active (depends on step 1).

11. Phase 3 - Runtime plused mode integration
12. Add plused-mode state in MmuGearBldc (same file) to represent edge phase, next-edge schedule, and active generation token for stale-callback suppression.
13. Route low-speed command path into plused scheduling when target RPM < (pulse_rpm - 100 RPM hysteresis offset) and tach control is active.
14. Keep all pin writes inside queue callback path only (_set_pin repeat/delay flow); no direct set_pwm or set_digital outside queue callback.
15. Stop/disable writes immediate PWM=0 from stop handler, then invalidates pending plused generation. No residual pulses after stop point.
16. Maintain continuous-mode logic unchanged for RPM >= pulse_rpm (depends on phase 2).

17. Phase 4 - Tach feedback behavior in plused mode
18. Update control eligibility and control loop so low-speed plused operation remains tach-governed rather than hard-disabled.
19. Apply tach RPM error correction to pulse ON-time (duty-cycle modulation only), not period or sync semantics or non-BLDC layers.
20. Add functional bounds to pulse ON-time (min/max clamp in phase 3 runtime) to avoid pathological always-on/always-off patterns.
21. Keep existing BLDC_CONTROL diagnostics meaningful by logging mode transitions and low-rate summaries rather than noisy per-edge spam (parallel with phase 3, finalized after phase 3).

22. Phase 5 - Tests and verification
23. Unit/behavior tests (parallelizable):
24. Extend Happy-Hare/.github/skills/run-test-rig/tests/run_test_rig_helpers.py with pulse-aware parsing/assert helpers only if existing helpers cannot assert fallback and stop behavior.
25. Extend Happy-Hare/.github/skills/run-test-rig/tests/test_run_test_rig_bldc.py with scenarios:
26. Low-speed command enters plused mode.
27. Plused mode exits correctly when speed rises above threshold.
28. Stop command terminates pulse activity cleanly.
29. Tach feedback still influences low-speed plused behavior.
30. Regression checks (depends on implementation + tests):
31. Python syntax check for Happy-Hare/extras/mmu/mmu_gear_bldc.py.
32. Targeted pytest run for BLDC scenarios and helper regressions.
33. Optional hardware test-rig validation pass for low-speed MOTOR=gear and low-speed synchronized move, then inspect klippy.log for fallback activation, bounded pulse behavior, and clean stop.

**Relevant files**
- Happy-Hare/extras/mmu/mmu_gear_bldc.py — primary implementation: config parse, fallback decision, queued pulse scheduling, tach interaction.
- Happy-Hare/.github/skills/run-test-rig/tests/test_run_test_rig_bldc.py — scenario-level behavior coverage for fallback lifecycle.
- Happy-Hare/.github/skills/run-test-rig/tests/run_test_rig_helpers.py — helper assertions/parsers for pulse fallback evidence in logs.
- Happy-Hare/config/base/mmu_hardware.cfg — template comments/examples for new short keys if user-facing config docs need update.
- Happy-Hare/printer_data/base/mmu_hardware.cfg — live config mirror update only if this environment expects synchronized comments/default examples.

**Verification**
1. Static check: compile Happy-Hare/extras/mmu/mmu_gear_bldc.py.
2. Pytest: run targeted BLDC scenario tests plus helper tests covering pulse fallback and stop behavior.
3. Log-level acceptance:
4. For target RPM < (pulse_rpm - 100), fallback enters plused mode.
5. For target RPM >= pulse_rpm, control exits pulsed, uses continuous path.
6. Stop/disable writes PWM=0 immediately; no residual pulses in log after stop.
7. Tach feedback evidence remains present during low-speed plused operation.
8. No regressions in gear+extruder and extruder+gear concurrency semantics.

**Decisions**
- Included: auto fallback below min-speed threshold using short keys pulse_rpm and pulse_ms.
- Included: implementation confined to BLDC module ownership in Happy-Hare/extras/mmu/mmu_gear_bldc.py.
- Included: queued pin-write contract preserved for all pulse edges.
- Excluded: changes to Happy-Hare/extras/mmu/mmu.py orchestration, sync-mode contract rewrites, and Klipper core edits.

**Further Considerations**
1. Control floor is `CONTROL_MIN_RPM = 150 RPM` (existing constant). Set default pulse_rpm >= 150 for predictable overlay.
2. Hysteresis: enter plused at (pulse_rpm - 100), exit at (pulse_rpm). Prevents mode chatter.
3. If low-speed oscillation appears, apply ON-time clamp before adding user keys.
4. Log one concise mode-transition per entry/exit; avoid per-edge spam for test robustness.