import types


class _Dummy:
    pass


class _DebugStepperMovement:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False


class _DummyHomingMove:
    def __init__(self, _printer, _endstops, toolhead):
        self._toolhead = toolhead

    def homing_move(self, pos, _speed, **_kwargs):
        self._toolhead.set_position(pos)
        return list(pos)


class _FakeSelector:
    FILAMENT_DRIVE_STATE = 1
    FILAMENT_RELEASE_STATE = 0

    def __init__(self):
        self._state = self.FILAMENT_DRIVE_STATE
        self.selected_gate = 0

    def filament_drive(self):
        self._state = self.FILAMENT_DRIVE_STATE

    def filament_release(self):
        self._state = self.FILAMENT_RELEASE_STATE

    def select_gate(self, gate):
        self.selected_gate = gate

    def restore_gate(self, gate):
        self.selected_gate = gate

    def has_bypass(self):
        return False

    def get_filament_grip_state(self):
        return self._state


class _DummyGCodeMove:
    def __init__(self):
        self.saved_states = {}

    def get_status(self, _eventtime):
        return {
            "speed_factor": 1.0,
            "extrude_factor": 1.0,
            "gcode_position": [0.0, 0.0, 0.0, 0.0],
        }

    def _save_state(self, name):
        self.saved_states[name] = {
            "speed_factor": 1.0,
            "extrude_factor": 1.0,
            "last_position": [0.0, 0.0, 0.0, 0.0],
        }


class _DummyGCode:
    def __init__(self, gcode_move=None):
        self.commands = {}
        self.ready_gcode_handlers = {}
        self.gcode_help = {}
        self._gcode_move = gcode_move
        self.raw_responses = []
        self.info_responses = []

    def register_command(self, name, handler, desc=None):
        prev = self.commands.get(name)
        if handler is None:
            self.commands.pop(name, None)
            self.ready_gcode_handlers.pop(name, None)
            self.gcode_help.pop(name, None)
            return prev

        self.commands[name] = handler
        self.ready_gcode_handlers[name] = handler
        if desc is not None:
            self.gcode_help[name] = desc
        return prev

    def run_script_from_command(self, command):
        if command.startswith("SAVE_GCODE_STATE NAME=") and self._gcode_move is not None:
            self._gcode_move._save_state(command.split("=", 1)[1])
        return None

    def respond_raw(self, message):
        self.raw_responses.append(message)

    def respond_info(self, message):
        self.info_responses.append(message)


class _DummyIdleTimeout:
    def __init__(self):
        self.idle_timeout = 600


class _DummyLedManager:
    def __init__(self, mmu):
        self.mmu = mmu

    def print_state_changed(self, *_args, **_kwargs):
        return None

    def action_changed(self, *_args, **_kwargs):
        return None

    def gate_map_changed(self, *_args, **_kwargs):
        return None


class _DummySensorManager:
    def __init__(self, mmu):
        self.mmu = mmu
        self.loaded_gates = set()

    def reset_active_unit(self, *_args, **_kwargs):
        return None

    def get_status(self, _eventtime):
        return {}

    def get_all_sensors(self, inactive=False):
        _ = inactive
        return {}

    def has_sensor(self, _name):
        return True

    def has_gate_sensor(self, _prefix, _gate):
        return False

    def check_gate_sensor(self, _prefix, _gate):
        return None

    def get_gate_sensor_name(self, prefix, gate):
        return "%s_%d" % (prefix, gate)

    def get_mapped_endstop_name(self, name):
        return name

    def check_sensor(self, name):
        if name == self.mmu.SENSOR_GATE:
            return self.mmu.gate_selected in self.loaded_gates
        if name.startswith(self.mmu.SENSOR_GEAR_PREFIX):
            try:
                gate = int(name.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                return False
            return gate in self.loaded_gates
        return False

    def disable_runout(self, *_args, **_kwargs):
        return None

    def enable_runout(self, *_args, **_kwargs):
        return None

    def confirm_loaded(self):
        return None


class _DummySyncFeedbackManager:
    def __init__(self, mmu):
        self.mmu = mmu
        self.flowguard_encoder_mode = 0

    def is_enabled(self):
        return False

    def is_active(self):
        return False

    def activate_flowguard(self, _eventtime):
        return None

    def deactivate_flowguard(self, _eventtime):
        return None

    def wipe_telemetry_logs(self):
        return None

    def get_sync_feedback_string(self, detail=False):
        _ = detail
        return "inactive"


class _DummyEnvironmentManager:
    def __init__(self, mmu):
        self.mmu = mmu


class _DummyCalibrationManager:
    def __init__(self, mmu):
        self.mmu = mmu

    def get_bowden_length(self):
        return 0.0


def _dummy_mmu_test(_mmu):
    return None


class _DummyPauseResume:
    def __init__(self):
        self.is_paused = False


class _DummySaveVariables:
    def __init__(self):
        self.allVariables = {"mmu__revision": 0}


class _DummyExtruderHeater:
    def __init__(self):
        self.target_temp = 0.0
        self.min_extrude_temp = 170.0


class _DummyExtruderObj:
    def __init__(self):
        self.heater = _DummyExtruderHeater()

    def get_status(self, _eventtime):
        return {"temperature": 25.0}

    def get_heater(self):
        return self.heater

    def process_move(self, print_time, move, ea_index):
        _ = print_time, move, ea_index
        return None


class _DummyRail:
    def __init__(self):
        self.steppers = []

    def get_extra_endstop(self, _name):
        return []

    def get_endstops(self):
        return []

    def is_endstop_virtual(self, _name):
        return False


class _DummyMmuKinematics:
    def __init__(self):
        self.rails = [None, _DummyRail()]
        self.accel_limit = None

    def set_accel_limit(self, accel):
        self.accel_limit = accel


class _DummyMmuToolhead:
    EXTRUDER_SYNCED_TO_GEAR = 1
    EXTRUDER_ONLY_ON_GEAR = 2
    GEAR_ONLY = 3
    GEAR_SYNCED_TO_EXTRUDER = 4

    def __init__(self, reactor=None):
        self._reactor = reactor
        self.sync_mode = None
        self._position = [0.0, 0.0, 0.0, 0.0]
        self._kinematics = _DummyMmuKinematics()
        self.mmu_extruder_stepper = types.SimpleNamespace(stepper=_DummyStepperRailBase._StepperObj())
        self.motion_events = []

    def _record_motion(self, operation, position):
        eventtime = 0.0
        if self._reactor is not None:
            eventtime = float(self._reactor.monotonic())
        old_y = self._position[1]
        new_y = float(position[1])
        self.motion_events.append(
            {
                "operation": operation,
                "eventtime": eventtime,
                "old_y": old_y,
                "new_y": new_y,
                "delta_y": new_y - old_y,
                "sync_mode": self.sync_mode,
            }
        )

    def get_position(self):
        return list(self._position)

    def set_position(self, position):
        self._record_motion("set_position", position)
        self._position = list(position)

    def move(self, position, _speed):
        self._record_motion("move", position)
        self._position = list(position)

    def flush_step_generation(self):
        return None

    def wait_moves(self):
        return None

    def dwell(self, _dwell):
        return None

    def quiesce(self):
        return None

    def sync(self, mode):
        self.sync_mode = mode

    def get_kinematics(self):
        return self._kinematics


class _DummyToolhead:
    def __init__(self):
        self._position = [0.0, 0.0, 0.0, 0.0]
        self.max_accel = 5000.0
        self._extruder = _DummyExtruderObj()

    def get_position(self):
        return list(self._position)

    def move(self, position, _speed):
        self._position = list(position)

    def flush_step_generation(self):
        return None

    def wait_moves(self):
        return None

    def dwell(self, _dwell):
        return None

    def get_last_move_time(self):
        return 0.0

    def get_status(self, _eventtime):
        return {"homed_axes": "xyz"}

    def get_extruder(self):
        return self._extruder


class _DummyStepperRailBase:
    class _StepperObj:
        def get_name(self):
            return "dummy_stepper"

        def set_trapq(self, _trapq):
            return None

        def generate_steps(self):
            return None

        def get_trapq(self):
            return None

        def get_commanded_position(self):
            return 0.0

        def get_mcu_position(self):
            return 0.0

        def get_rotation_distance(self):
            return (1.0, 200)

        def get_step_dist(self):
            return 0.01

    def __init__(self, config, **_kwargs):
        self.config = config
        self.endstops = []
        self.steppers = [self._StepperObj()]

    def get_steppers(self):
        return list(self.steppers)

    def add_stepper_from_config(self, _config, **_kwargs):
        return None

    def add_extra_stepper(self, _config, **_kwargs):
        return None

    def setup_itersolve(self, *_args):
        return None

    def get_range(self):
        return (0.0, 100.0)

    def get_homing_info(self):
        return types.SimpleNamespace(
            position_endstop=0.0,
            positive_dir=True,
            retract_dist=0.0,
            speed=10.0,
            retract_speed=5.0,
            second_homing_speed=5.0,
        )

    def set_position(self, _newpos):
        return None

    def is_endstop_virtual(self, _name):
        return False

    def get_extra_endstop(self, _name):
        return None

    def get_endstops(self):
        return []


class _DummyMoveQueue:
    def __init__(self, _toolhead):
        self.flush_time = 0.0

    def set_flush_time(self, flush_time):
        self.flush_time = flush_time


class _DummyMachine:
    def __init__(self):
        self.num_gates = 2
        self.num_units = 1
        self.gate_counts = [2]
        self.use_bldc_gear = False
        self.use_stepper_gear = True
        self.homing_extruder = False
        self.selector_type = "VirtualSelector"
        self.filament_always_gripped = False
        self.require_bowden_move = True
        self.variable_bowden_lengths = False
        self.variable_rotation_distances = False
        self.multigear = False
        self.units = [types.SimpleNamespace(unit_index=0, index=0, first_gate=0, num_gates=2, name="unit0")]

    def unit_uses_bldc(self, _unit_index):
        return self.use_bldc_gear

    def get_bldc_section_names_for_unit(self, unit):
        return [
            "mmu_gear_bldc",
            "mmu_gear_bldc %s" % unit.name,
            "mmu_gear_bldc unit%d" % (unit.unit_index + 1),
        ]

    def get_mmu_unit_by_gate(self, gate):
        for unit in self.units:
            if unit.first_gate <= gate < unit.first_gate + unit.num_gates:
                return unit
        return self.units[0]


class _DummyMcuObj:
    def __init__(self, name="mcu"):
        self._name = name

    def is_fileoutput(self):
        return False

    def get_name(self):
        return self._name


class _MiniMutex:
    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False


class _MiniReactor:
    def mutex(self):
        return _MiniMutex()


class _MiniPrinter:
    def get_start_args(self):
        return {}

    def register_event_handler(self, _name, _cb):
        return None

    def get_reactor(self):
        return _MiniReactor()

    def config_error(self, message):
        return RuntimeError(message)

    def send_event(self, *_args):
        return None

    def invoke_shutdown(self, *_args):
        return None

    def request_exit(self, *_args):
        return None
