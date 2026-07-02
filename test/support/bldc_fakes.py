import importlib.util
import os
import sys
import types


class _DummyMotionQueuing:
    def register_flush_callback(self, _callback):
        return None

    def note_mcu_movequeue_activity(self, _print_time, is_step_gen=False):
        _ = is_step_gen
        return None


def _resolve_klipper_output_pin_path(repo_root):
    candidates = [
        os.path.abspath(os.path.join(repo_root, "..", "..", "..", "klipper", "klippy", "extras", "output_pin.py")),
        os.path.abspath(os.path.join(repo_root, "..", "..", "klipper", "klippy", "extras", "output_pin.py")),
        os.path.abspath(os.path.join(repo_root, "..", "klipper", "klippy", "extras", "output_pin.py")),
        os.path.abspath(os.path.join("c:/git/klipper/klippy/extras/output_pin.py")),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise RuntimeError("Unable to locate klipper/klippy/extras/output_pin.py")


def install_output_pin_module(repo_root):
    display_module = types.ModuleType("extras.display")

    class _DisplayStub:
        @staticmethod
        def lookup_display_templates(_config):
            return types.SimpleNamespace(get_display_templates=lambda: {})

    display_module.display = _DisplayStub()
    sys.modules["extras.display"] = display_module

    module_path = _resolve_klipper_output_pin_path(repo_root)
    spec = importlib.util.spec_from_file_location("extras.output_pin", module_path)
    output_pin_module = importlib.util.module_from_spec(spec)
    sys.modules["extras.output_pin"] = output_pin_module
    spec.loader.exec_module(output_pin_module)
    return output_pin_module


class FakeMcu:
    def __init__(self, min_schedule=0.001, reactor=None, delivery_latency=None):
        self._min_schedule = float(min_schedule)
        self._reactor = reactor
        # Simulate host->MCU delivery delay before pin commands are applied.
        self._delivery_latency = float(min_schedule if delivery_latency is None else delivery_latency)

    def current_eventtime(self):
        if self._reactor is None:
            return 0.0
        return float(self._reactor.monotonic())

    def min_schedule_time(self):
        return self._min_schedule

    def delivery_latency(self):
        return self._delivery_latency

    def estimated_print_time(self, eventtime):
        return float(eventtime)


class StrictMcu:
    def __init__(self, *, min_schedule=0.1, base_print_time=5.0, reactor=None, delivery_latency=None):
        self._min_schedule = float(min_schedule)
        self._base_print_time = float(base_print_time)
        self._reactor = reactor
        self._calls = 0
        # Keep strict model closer to hardware by accounting for dispatch delay.
        self._delivery_latency = float(min_schedule if delivery_latency is None else delivery_latency)

    def current_eventtime(self):
        if self._reactor is None:
            return 0.0
        return float(self._reactor.monotonic())

    def min_schedule_time(self):
        return self._min_schedule

    def delivery_latency(self):
        return self._delivery_latency

    def estimated_print_time(self, eventtime):
        self._calls += 1
        return self._base_print_time + float(eventtime)


class FakePin:
    def __init__(self, pin_name="fake_pin", mcu=None):
        self._pin = pin_name
        self._mcu = mcu or FakeMcu()
        self.digital_calls = []
        self.pwm_calls = []
        self.pwm_events = []

    def setup_max_duration(self, _value):
        return None

    def setup_start_value(self, _start, _shutdown):
        return None

    def setup_cycle_time(self, _cycle_time, _hardware_pwm):
        return None

    def _validate_schedule(self, print_time):
        if print_time is None:
            return
        eventtime = 0.0
        if hasattr(self._mcu, "current_eventtime"):
            eventtime = self._mcu.current_eventtime()
        delivery_latency = 0.0
        if hasattr(self._mcu, "delivery_latency"):
            delivery_latency = self._mcu.delivery_latency()
        floor_eventtime = eventtime + delivery_latency
        floor = self._mcu.estimated_print_time(floor_eventtime) + self._mcu.min_schedule_time()
        if print_time < floor - 1e-9:
            raise RuntimeError(f"Timer too close: {floor - print_time:.2f} over")

    def set_pwm(self, print_time, value):
        self._validate_schedule(print_time)
        eventtime = 0.0
        if hasattr(self._mcu, "current_eventtime"):
            eventtime = float(self._mcu.current_eventtime())
        self.pwm_calls.append((print_time, value))
        self.pwm_events.append({"eventtime": eventtime, "print_time": print_time, "value": value})

    def set_digital(self, print_time, value):
        self._validate_schedule(print_time)
        self.digital_calls.append((print_time, value))

    def get_mcu(self):
        return self._mcu


class FakeQueuedPwm(FakePin):
    def __init__(self, config, pin_params):
        pin_name = pin_params.get("pin", "pwm_pin")
        super().__init__(pin_name=pin_name, mcu=pin_params.get("mcu"))
        self.config = config
        self.pin_params = pin_params


class FakeCounter:
    def __init__(self, _printer, pin, sample_time, poll_interval):
        self.pin = pin
        self.sample_time = sample_time
        self.poll_interval = poll_interval
        self.callback = None

    def setup_callback(self, callback):
        self.callback = callback

    def emit(self, event_time, count, count_time):
        if self.callback is not None:
            self.callback(event_time, count, count_time)


class FakeReactor:
    NOW = 0.0
    NEVER = 1.0e30

    def __init__(self):
        self._now = 0.0
        self.timers = []

    def _run_due_timers(self):
        # Single-pass servicing avoids runaway spin when callbacks reschedule at NOW.
        due = [timer for timer in self.timers if timer["when"] <= self._now]
        for timer in due:
            if timer["when"] > self._now:
                continue
            next_when = timer["callback"](self._now)
            timer["when"] = self.NEVER if next_when is None else float(next_when)

    def monotonic(self):
        return self._now

    def pause(self, wake_time):
        self._now = float(wake_time)
        self._run_due_timers()
        return self._now

    def register_timer(self, callback, when=None):
        timer = {"callback": callback, "when": self.NEVER if when is None else when}
        self.timers.append(timer)
        return timer

    def update_timer(self, timer, when):
        timer["when"] = float(when)
        if timer["when"] <= self._now:
            self._run_due_timers()


class FakePins:
    def __init__(self, reactor=None):
        self._mcu = FakeMcu(reactor=reactor)
        self.last_digital_pin = None

    def setup_pin(self, mode, pin_name):
        if mode != "digital_out":
            raise RuntimeError("Unexpected mode %s" % mode)
        self.last_digital_pin = FakePin(pin_name=pin_name, mcu=self._mcu)
        return self.last_digital_pin

    def lookup_pin(self, pin_name, can_invert=True):
        return {"pin": pin_name, "can_invert": can_invert, "mcu": self._mcu}


class FakeExtruder:
    def __init__(self):
        self.calls = []

    def process_move(self, print_time, move, ea_index):
        self.calls.append((print_time, move, ea_index))


class FakeToolhead:
    def __init__(self, extruder=None):
        self._extruder = extruder or FakeExtruder()
        self.flush_count = 0

    def get_extruder(self):
        return self._extruder

    def flush_step_generation(self):
        self.flush_count += 1


class FakePrinter:
    def __init__(self, reactor=None, pins=None, toolhead=None):
        self._reactor = reactor or FakeReactor()
        self._pins = pins or FakePins(reactor=self._reactor)
        self._toolhead = toolhead or FakeToolhead()
        self._motion_queuing = _DummyMotionQueuing()
        self.handlers = []

    def get_reactor(self):
        return self._reactor

    def register_event_handler(self, name, callback):
        self.handlers.append((name, callback))

    def load_object(self, _config, name):
        if name == "motion_queuing":
            return self._motion_queuing
        return None

    def lookup_object(self, name, default=None):
        if name == "pins":
            return self._pins
        if name == "toolhead":
            return self._toolhead
        return default


class FakeConfig:
    def __init__(self, printer, values=None, name="mmu_gear_bldc"):
        self._printer = printer
        self._values = values or {}
        self._name = name

    def get_printer(self):
        return self._printer

    def get_name(self):
        return self._name

    def get(self, key, default=None):
        return self._values.get(key, default)

    def getboolean(self, key, default=False):
        return bool(self._values.get(key, default))

    def getint(self, key, default=0, minval=None, maxval=None):
        value = int(self._values.get(key, default))
        if minval is not None and value < minval:
            raise RuntimeError("%s below min" % key)
        if maxval is not None and value > maxval:
            raise RuntimeError("%s above max" % key)
        return value

    def getfloat(self, key, default=0.0, minval=None, maxval=None, above=None):
        value = float(self._values.get(key, default))
        if minval is not None and value < minval:
            raise RuntimeError("%s below min" % key)
        if maxval is not None and value > maxval:
            raise RuntimeError("%s above max" % key)
        if above is not None and value <= above:
            raise RuntimeError("%s must be above threshold" % key)
        return value

    def getintlist(self, key, default=()):
        raw = self._values.get(key, default)
        return [int(v) for v in raw]

    def error(self, message):
        return RuntimeError(message)


class FakeMmu:
    def __init__(self, toolhead=None, gate_selected=0):
        self.toolhead = toolhead or FakeToolhead()
        self.gate_selected = gate_selected
        self.stepper_logs = []
        self.warning_logs = []
        self.movequeues_wait_calls = 0

    def log_stepper(self, message):
        self.stepper_logs.append(message)

    def log_warning(self, message):
        self.warning_logs.append(message)

    def movequeues_wait(self):
        self.movequeues_wait_calls += 1


class FakeMove:
    def __init__(
        self,
        axes_r,
        start_v=0.0,
        cruise_v=0.0,
        end_v=0.0,
        accel=1.0,
        accel_t=0.1,
        cruise_t=0.1,
        decel_t=0.1,
    ):
        self.axes_r = axes_r
        self.start_v = start_v
        self.cruise_v = cruise_v
        self.end_v = end_v
        self.accel = accel
        self.accel_t = accel_t
        self.cruise_t = cruise_t
        self.decel_t = decel_t


def load_mmu_gear_bldc_module(repo_root):
    extras_dir = os.path.join(repo_root, "extras")
    mmu_dir = os.path.join(extras_dir, "mmu")
    module_path = os.path.join(mmu_dir, "mmu_gear_bldc.py")

    for key in [
        "extras.mmu.mmu_gear_bldc",
        "extras.mmu.mmu_shared",
        "extras.display",
        "extras.mmu",
        "extras.pulse_counter",
        "extras.pwm_tool",
        "extras.output_pin",
        "extras",
    ]:
        sys.modules.pop(key, None)

    extras_pkg = types.ModuleType("extras")
    extras_pkg.__path__ = [extras_dir]
    sys.modules["extras"] = extras_pkg

    mmu_pkg = types.ModuleType("extras.mmu")
    mmu_pkg.__path__ = [mmu_dir]
    sys.modules["extras.mmu"] = mmu_pkg

    mmu_shared = types.ModuleType("extras.mmu.mmu_shared")

    class MmuError(Exception):
        pass

    mmu_shared.MmuError = MmuError
    sys.modules["extras.mmu.mmu_shared"] = mmu_shared

    pulse_counter_module = types.ModuleType("extras.pulse_counter")
    pulse_counter_module.MCU_counter = FakeCounter
    sys.modules["extras.pulse_counter"] = pulse_counter_module

    pwm_tool_module = types.ModuleType("extras.pwm_tool")
    pwm_tool_module.MCU_queued_pwm = FakeQueuedPwm
    sys.modules["extras.pwm_tool"] = pwm_tool_module

    install_output_pin_module(repo_root)

    spec = importlib.util.spec_from_file_location("extras.mmu.mmu_gear_bldc", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["extras.mmu.mmu_gear_bldc"] = module
    spec.loader.exec_module(module)
    return module
