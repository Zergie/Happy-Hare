---
description: "Happy Hare Python coding style for extras/**/*.py. Covers file headers, class layout, comments, logging, imports, string formatting, Klipper patterns (config reading, gcode commands, get_status, event handlers), and naming conventions."
applyTo: "extras/**/*.py"
---

# Happy Hare Python Style

## File Header

Every file starts with this exact block (copyright year range updated as needed):

```python
# Happy Hare MMU Software
#
# <one-line purpose>
#
# Copyright (C) 2022-2026  moggieuk#6538 (discord)
#                          moggieuk@hotmail.com
#
# (\_/)
# ( *,*)
# (")_(") Happy Hare Ready
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#
```

## Module-Level Constants

UPPER_CASE. Add inline comment when meaning is not obvious.

```python
MAX_SCHEDULE_TIME = 5.0
CHECK_INTERVAL = 0.5  # How often to check extruder movement (seconds)
```

## Class Layout Order

1. Module-level constants
2. `__init__`
3. `_handle_*` event handlers (klippy events)
4. Public API methods
5. Private helpers (`_` prefix)
6. `get_status`
7. Nested helper classes (last)

## `__init__` Style

- Group `self.x` assignments by purpose with a one-line section comment.
- Read all config values in a single logical block. Never scatter `config.getXxx` calls across methods.
- Inline `# Key: X, Value: Y` comments on dicts.

```python
self.motor_mcu_pins = {}  # Key: pin_name, Value: mcu_pin
self.last_value = {}      # Key: pin_name, Value: Last pwm value
```

## Section Dividers

Use long-dash separators to group related methods inside a class:

```python
# Change operation in progress and DC motor PWM control -------------------
```

## Docstrings

Short imperative-mood sentence on the first line. List key behaviors with `-` bullets only when needed. No Google/NumPy style, no `Args:` / `Returns:` blocks.

```python
def enable(self):
    """
    Globally enable monitoring and start the watchdog immediately.
    """
```

## Inline Comments

- Explain *why*, not *what*.
- Trailing inline: one or two spaces before `#`.
- Acceptable one-liner when body is obvious: `if pin == '': return True`

## Logging

Use `self.mmu.log_trace / log_debug / log_error`. Prefix messages with the module name in caps followed by `:`.

```python
self.mmu.log_trace("ESPOOLER: Trigger fired for gate %d, state=%s" % (gate, state))
self.mmu.log_error("BLDC: Unexpected state %s" % state)
```

Never raise exceptions that escape public methods — log and return/ignore instead.

## String Formatting

Use `%` formatting throughout (Python 2.7 compatibility). No f-strings.

```python
"Gate %d set to %s (pwm: %.2f)" % (gate, operation, value)
```

## Deferred Imports

Import from sibling modules inside methods to avoid circular imports. Add a short comment.

```python
from .mmu import Mmu  # For operation names
```

## Klipper Patterns

### Config reading (in `__init__`)

```python
self.is_pwm = config.getboolean("pwm", True)
self.cycle_time = config.getfloat("cycle_time", 0.100, above=0., maxval=MAX_SCHEDULE_TIME)
```

### Event handlers

```python
self.printer.register_event_handler("mmu:disabled", self._handle_mmu_disabled)
```

Handler names: `_handle_<event_slug>`.

### GCode commands

Declare help string as class attribute immediately before the method. Register via `register_mux_command`.

```python
cmd_SET_SERVO_help = "Set servo angle"
def cmd_SET_SERVO(self, gcmd):
    ...
```

### `get_status`

Always return a plain dict. Required for Klipper status interface.

```python
def get_status(self, eventtime):
    return {'value': self.last_value}
```

## Naming

- `snake_case` for functions and variables.
- `PascalCase` for classes.
- `_single_leading_underscore` for private/internal helpers.
- `UPPER_CASE` for module-level constants.
- Prefix log messages with the subsystem name in ALL CAPS: `"ESPOOLER: ..."`, `"BLDC: ..."`.
- Gate/pin dicts keyed by descriptive string names, not raw integers, when the key has semantic meaning.

## What to Avoid

- f-strings.
- Google/NumPy docstring style (`Args:`, `Returns:`).
- Raising unhandled exceptions from public methods — log and return.
- Scattering config reads outside `__init__`.
- Duplicate `GCodeRequestQueue` instances per MCU (reuse existing queue).
