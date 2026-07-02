# MMU_LOAD

`MMU_LOAD` loads filament from the currently selected gate into the extruder and, in the normal MMU path, on to the nozzle. It is the direct load operation for manual recovery, bypass use, and standalone loading outside a toolchange.

## Where it fits

The command is registered in `extras/mmu/mmu.py` and is described in the public command reference and basic operation wiki pages. The load path itself is implemented by `cmd_MMU_LOAD()` and the shared `load_sequence()` helper. If `gcode_load_sequence: 1` is enabled, Happy Hare delegates the load to `_MMU_LOAD_SEQUENCE` instead of using the built-in sequence.

Related source/context:

- `extras/mmu/mmu.py` for the command handler and load sequencing
- `Happy-Hare.wiki/Command-Reference.md` for the public command summary
- `Happy-Hare.wiki/Basic-Operation.md` for the user-facing load flow
- `Happy-Hare.wiki/Custom-Load-Unload-Sequences.md` for the optional macro hook

## Command surface

```text
MMU_LOAD
MMU_LOAD SKIP_PURGE=1
MMU_LOAD EXTRUDER_ONLY=1
```

Documented parameters:

- `SKIP_PURGE=1` skips the post-load standalone purge step.
- `EXTRUDER_ONLY=1` loads only the extruder path instead of running the full gate/bowden load.

Implementation note:

- `RESTORE=0` is also accepted by the handler, even though it is not listed in the public command table. It suppresses the normal post-load restore/parking cleanup used by the load flow.

## How It Runs

`MMU_LOAD` first requires the MMU to be enabled, homed, and calibrated for the selected gate. If filament is already loaded, the command logs that fact and exits without moving anything.

From there the command takes one of three paths:

| Path | Trigger | What runs | What to expect |
| --- | --- | --- | --- |
| Full MMU load | Default case with a normal gate selected | Shared load sequence through gate, bowden, and extruder/toolhead stages | Filament is pulled from the selected gate, moved through the bowden, then loaded into the extruder and on to the nozzle using the configured sensor strategy |
| Extruder-only load | `EXTRUDER_ONLY=1`, or automatically when the bypass selector is active | Extruder-only branch of the load helper | Filament is advanced only through the extruder/toolhead path; the gate and bowden stages are skipped |
| Macro-defined load | `gcode_load_sequence: 1` | `_MMU_LOAD_SEQUENCE` / configured load macro | The command hands control to the macro with the current filament position and load context |

## Physical Behavior

In the normal MMU path, the load starts from the selected gate, then advances through the bowden, then finishes by homing or syncing through the extruder/toolhead section depending on the configured sensors. The exact sensor sequence depends on your hardware: encoder-based gate confirmation, gate homing, extruder-entry homing, and toolhead-sensor homing are all handled by the shared load code.

If the bypass selector is active, the command behaves like a direct extruder load. That is the intended path for bypass printing and recovery use cases where filament is not being routed through a gate and bowden segment.

## Purge And Post-Load Behavior

By default the command uses Happy Hare’s standalone purge path after a successful load. That makes `MMU_LOAD` a practical standalone recovery command, not just a transport command. Use `SKIP_PURGE=1` when you only want filament positioned for later manual or macro-driven use.

The load command also participates in the normal load-operation parking/restore logic used by Happy Hare’s toolhead movement layer. If you need to suppress that cleanup for a custom workflow, the handler accepts `RESTORE=0`.

## Defaults, Constraints, And Gotchas

- `MMU_LOAD` is not a generic “move filament anywhere” command; it assumes a valid selected tool/gate unless bypass is active.
- The command is blocked if the MMU is disabled, not homed, or not calibrated for the selected gate.
- If the current filament position is already loaded, the command exits early instead of trying to duplicate the load.
- The full load path is the default. `EXTRUDER_ONLY=1` is the exception, not the normal case.
- When `gcode_load_sequence` is enabled, the macro receives the current filament position plus the requested load context, so custom sequences must preserve the filament state machine correctly.

## Sequence Macro Contract

When the macro-based load path is enabled, Happy Hare calls the configured load macro with these fields:

- `FILAMENT_POS`
- `LENGTH`
- `FULL`
- `HOME_EXTRUDER`
- `SKIP_EXTRUDER`
- `EXTRUDER_ONLY`

That hook is the supported way to replace the built-in load logic for unusual MMU hardware or custom load choreography.

## Expected Outcome

After a successful `MMU_LOAD`, Happy Hare should report filament loaded, update its internal filament-position state, and leave the MMU ready for purge or print continuation. In a bypass load, the result is only an extruder load; in a full MMU load, the result is a fully loaded filament path through the gate, bowden, extruder, and nozzle.
