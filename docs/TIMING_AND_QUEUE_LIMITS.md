# Timing and Queue Limits

Last verified against source and test-rig behavior: 2026-07-04.

This note captures practical timing limits for Happy Hare code that schedules MCU pin or motion-related updates from Python.

## Timing layers

Happy Hare code normally deals with several different clocks and queues:

- Host/reactor time: Python timers and callbacks.
- Print time: MCU-scheduled time used for pin writes and motion events.
- Motion queue time: planned move windows and lookahead/flush behavior.
- MCU command queue time: minimum safe spacing for commands accepted by the MCU queue path.

These layers are related, but they are not interchangeable. A Python timer can wake frequently without proving that MCU pin commands can be delivered at the same rate.

## Queue interval is the hard limit

For queued pin updates, the effective update rate is limited by the queue implementation and the MCU scheduling floor. On the current test rig, the practical queue interval is about `0.100s`.

Config values and local timers may request shorter intervals, but those requests cannot make the MCU command queue drain faster. If code creates commands faster than the queue can flush them, the extra commands accumulate as future work.

## Backlog failure mode

Producing queued pin writes faster than the queue drain rate can cause delayed shutdown behavior:

1. Motion ends or a stop is requested.
2. The stop write is queued behind already-scheduled future writes.
3. Hardware continues receiving older nonzero writes until the backlog drains.
4. The final zero/off write arrives late.

This can make one actuator continue after the component it was meant to follow has stopped, even when the high-level motion state has already ended.

## Design rule

Do not schedule host-side queued pin writes faster than the proven queue drain interval for that path.

When adding a configurable update interval:

- Clamp it to the queue's effective minimum interval.
- Log a warning when the requested interval is below the queue floor.
- Keep stop/off writes on the same queued path, but avoid building backlog ahead of them.
- Treat `mcu.min_schedule_time()` and queue behavior as safety constraints, not tuning suggestions.

## When faster updates are required

If behavior genuinely requires updates faster than the host queued pin path can provide, do not solve it by shortening a Python timer alone.

Use a mechanism that runs at the needed timing layer, such as:

- MCU-side scheduling designed for that update rate.
- A dedicated firmware/driver feature.
- A hardware controller that owns the fast control loop.
- A lower-frequency host loop that only commands average behavior.

## Log clues

Useful signs that the host is outrunning the queue:

- Queued pin writes keep appearing after the associated motion window has expired.
- Last nonzero pin write is much later than the last active motion descriptor.
- Stop/off write appears only after a long tail of nonzero writes.
- Requested interval is shorter than observed `BLDC_SET_PIN` or similar pin-write spacing.

When this happens, fix the producer rate or queue mechanism before tuning higher-level control behavior.
