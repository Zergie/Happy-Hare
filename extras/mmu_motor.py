# Happy Hare MMU Software
#
# (\_/)
# ( *,*)
# (")_(") Happy Hare Ready
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#
import logging
import math
from . import pulse_counter, output_pin, heaters, pid_calibrate
from . import mmu_espooler, mmu


MAX_SCHEDULE_TIME = 5.0
HOMING_START_DELAY = 0.001
ENDSTOP_SAMPLE_TIME = .000015
ENDSTOP_SAMPLE_COUNT = 4

PID_PARAM_BASE = 255
MIN_SPEED = 0.0
MAX_SPEED = 200.0

MOTOR_START_DELAY = 0.5
MOTOR_WARMUP_DELAY = 0.05
MOTOR_RUN_DELAY = 2.0
MOTOR_STOP_DELAY = 1.0

# directions
FORWARD  = 1
BACKWARD = 0


# class ControlPID:
#     def __init__(self, gcrq):
#         self.gcrq = gcrq
#         self.max_power = 100.0
#         self.Kp = 50. / PID_PARAM_BASE
#         self.Ki = .2 / PID_PARAM_BASE
#         self.Kd = 9. / PID_PARAM_BASE
#         self.prev_speed_time = 0.
#         self.integral = 0.0
#         self.previous_error = 0.0

#     def reset(self):
#         self.integral = 0.0
#         self.previous_error = 0.0

#     def update(self, read_time, speed, target_speed):
#         time_diff = read_time - self.prev_speed_time

#         error = target_speed - speed

#         p_out = self.Kp * error

#         self.integral += error * time_diff
#         if self.Ki:
#             integ_max = self.max_power / self.Ki
#             self.integral = max(0., min(integ_max, self.integral))
#         i_out = self.Ki * self.integral

#         derivative = (error - self.previous_error) / time_diff
#         d_out = self.Kd * derivative

#         co = p_out + i_out + d_out
#         bounded_co = max(0., min(self.max_power, co))

#         # if target_speed > 0:
#         #     logging.info("mmu_motor pid: speed=%.1f target=%.1f deriv=%f err=%f integ=%f co=%f",
#         #         speed, target_speed, derivative, error, self.integral, co)
#         self.gcrq.send_async_request((bounded_co, None))

#         if target_speed != 0:
#             self.previous_error = error
#         else:
#             self.integral = 0.0
#             self.previous_error = 0.0

class MMUMotor():
    def __init__(self, config):
        self.name = config.get_name().split()[-1]
        self.printer = config.get_printer()
        self.reactor = self.printer.get_reactor()
        self.toolhead = None
        self.mmu = None
        self.endstops = []
        self.speed = 0
        self.accel = 0
        self.smooth_time = config.getfloat('smooth_time', 2., above=0.)
        self.inv_smooth_time = 1. / self.smooth_time
        self.pulses_to_power = {}

        ppins = self.printer.lookup_object('pins')

        # PWM pin
        self.pwm_pin = ppins.setup_pin("pwm", config.get('pwm_pin', None))
        hardware_pwm = config.getboolean("hardware_pwm", False)
        cycle_time = config.getfloat("cycle_time", 0.00005, above=0., maxval=MAX_SCHEDULE_TIME) # 20 khz
        self.pwm_pin.setup_cycle_time(cycle_time, hardware_pwm)
        self.pwm_pin.setup_max_duration(0.)
        self.pwm_pin.setup_start_value(0, 0)
        self.power = self.base_power = self.target_speed = 0.0

        # Direction pin
        self.dir_pin = ppins.setup_pin("digital_out", config.get('dir_pin', None))
        self.dir_pin.setup_max_duration(0.)
        self.dir_pin.setup_start_value(0, 0)
        self._forward = 0
        self.direction = self._forward

        # For counter functionality
        self.sample_time = config.getfloat('sample_time', 0.1, above=0.)
                                               # 0.00055555555556 <- based on This must be smaller than 30/(tachometer_ppr*rpm)
                                               # 0.00033333333333 <- based on fan.py 2ppr<9ppr
                                               # 0.0015 <- default in fan.py
        poll_time = config.getfloat('poll_time', 0.00025, above=0.)
        self._counter = pulse_counter.MCU_counter(self.printer, config.get('encoder_pin', None), self.sample_time, poll_time)
        self._counter.setup_callback(self._counter_callback)
        self.resolution = config.getfloat('encoder_resolution', 1., above=0.)
        self._last_time = None
        self._pulses = self._last_pulses = self._initial_encoder_position = 0
        self.smoothed_velocity = 0.0
        self._velocity = []

        # check that all pins are on the same mcu
        self.mcu = self.pwm_pin.get_mcu()
        if not (self.pwm_pin.get_mcu().get_name() ==
                self.dir_pin.get_mcu().get_name() ==
                self._counter._mcu.get_name()):
            raise config.error("All motor pins must be on the same MCU")

        # Setup control
        if hasattr(output_pin, 'GCodeRequestQueue'):
            self.gcrq = output_pin.GCodeRequestQueue(config, self.mcu, self._set_pin)
        else:
            self.gcrq = mmu_espooler.GCodeRequestQueue(config, self.mcu, self._set_pin)
        # self.control = ControlPID(self.gcrq)
        self.control = heaters.ControlPID(self, config)

        # G-code commands
        self.gcode = self.printer.lookup_object('gcode')
        self.gcode.register_mux_command("MMU_MOTOR", "MOTOR", self.name,
                                        self.cmd_MMU_MOTOR,
                                        desc=self.cmd_MMU_MOTOR_help)
        self.gcode.register_mux_command("MMU_MOTOR_CALIBRATE", "MOTOR", self.name,
                                        self.cmd_MMU_MOTOR_CALIBRATE,
                                        desc=self.cmd_MMU_MOTOR_CALIBRATE_help)

        # Register for connect event
        self.printer.register_event_handler('klippy:connect', self.handle_connect)

    def handle_connect(self):
        self.mmu = self.printer.lookup_object('mmu')
        self.toolhead = self.printer.lookup_object('toolhead')
        self.save_variables = self.printer.lookup_object('save_variables')
        self.mmu_toolhead = self.mmu.mmu_toolhead
        self.pulses_to_power = self.save_variables.allVariables.get(mmu.Mmu.VARS_MMU_MOTOR_POWER_MAP_PREFIX + self.name, {})

    # Callback for MCU_counter
    def _counter_callback(self, print_time, pulses, pulses_time):
        if self._last_time is None:  # First sample
            time_diff = print_time
            self._last_time = print_time
            new_pulses = pulses
        elif pulses_time > self._last_time:
            time_diff = pulses_time - self._last_time
            self._last_time = pulses_time
            new_pulses = pulses - self._last_pulses
        else:  # No counts since last sample
            time_diff = pulses_time - self._last_time
            self._last_time = print_time
            new_pulses = 0

        if new_pulses == 0:
            self._velocity.append((print_time, 0.0))

            if self.power == 0.0 and -0.1 < time_diff < 0.0:
                self.log("<----- Motor stopped. Total pulses: %d, Distance: %.3f mm, time_diff=%.6f s" % (
                    self.get_encoder_pulses(),
                    self.get_encoder_distance(),
                    time_diff))
        else:
            self._velocity.append((print_time, self.pulses_to_distance(new_pulses / time_diff)))
            self._pulses += new_pulses

            velocity = self._velocity[-1][1]
            velocity_diff = velocity - self.smoothed_velocity
            adj_time = min(time_diff * self.inv_smooth_time, 1.)
            self.smoothed_velocity += velocity_diff * adj_time

            # velocity = sum(self._velocity[-3:]) / len(self._velocity[-3:])

            # self.control.update(pulses_time, self.pulses_to_distance(velocity), self._target_speed)
            if self.target_speed > 1.0 and self.control is not None:
                # self.control.temperature_update(pulses_time, velocity, self.target_speed)
                self.control.temperature_update(print_time, velocity, self.target_speed)

            self.log("-----> _counter_callback(print_time=%f, pulses=%d, pulses_time=%.6f) => time_diff=%.6f s, velocity=%.3f mm/s, smoothed=%.3f mm/s" % (
                print_time, pulses,  pulses_time, time_diff, self.pulses_to_distance(velocity), self.pulses_to_distance(self.smoothed_velocity)))

        self._last_pulses = pulses

    # This is the actual callback method to update pin signal (pwm or digital)
    def _set_pin(self, print_time, action):
        pwm_value, dir_value = action

        self.log("-----> _set_pin(pwm=%.1f) @ print_time: %.8f" % (pwm_value, print_time))

        if pwm_value is not None:
            if self.power == 0.0 and pwm_value > 0.0:
                self.log("<----- Motor starting.")

            self.pwm_pin.set_pwm(print_time, pwm_value / 100.0)
            self.power = pwm_value

        if dir_value is not None:
            if self.direction != dir_value:
                self.dir_pin.set_digital(print_time, dir_value)

    # vv heater interface method
    def set_pwm(self, print_time, pid_value):
        correction = pid_value - 100.
        min_power = min(self.pulses_to_power.values()) if self.pulses_to_power else 0.
        power = max(min_power, min(self.base_power + correction, 100.))
        self.log("set_pwm: base_power=%.1f correction=%.1f => power=%.1f" % (self.base_power, correction, power))
        self.gcrq.send_async_request((power, self.direction))

    def get_pwm_delay(self):
        return 0.3

    def get_max_power(self):
        return 200.0

    def get_smooth_time(self):
        return self.smooth_time

    def set_control(self, control):
        old_control = self.control
        self.control = control
        self.target_speed = 0.
        return old_control

    def alter_target(self, target_speed):
        if target_speed:
            target_speed = max(MIN_SPEED, min(MAX_SPEED, target_speed))
        self.target_speed = target_speed
    # ^^ heater interface method

    def get_name(self):
        return self.name

    def get_status(self, eventtime):
        return {
            'forward': self._forward,
            'encoder_pulses': min(0, self._pulses),
            'encoder_distance': min(0, self.get_encoder_distance()),
            'target_speed': self.target_speed,
            'power': self.power,
            'direction': self.direction,
            'velocity_samples': self._velocity,
            'smoothed_velocity': self.smoothed_velocity,
            'velocity_to_power': self.pulses_to_power,
            'control_Kp': self.control.Kp * PID_PARAM_BASE if hasattr(self.control, 'Kp') else None,
            'control_Ki': self.control.Ki * PID_PARAM_BASE if hasattr(self.control, 'Ki') else None,
            'control_Kd': self.control.Kd * PID_PARAM_BASE if hasattr(self.control, 'Kd') else None,
        }

    def log(self, msg):
        if self.name == "unit0":
            logging.info("mmu_motor %s: %s" % (self.name, msg))

    def speed_to_power(self, speed):
        pulses = self.distance_to_pulses(speed)
        pulseband = (
            min([i for i in self.pulses_to_power.keys() if i > 0]),
            max(self.pulses_to_power.keys())
        )

        if pulses < pulseband[0]:
            raise self.gcode.error("target speed too low, no suitable power found. (min %.1f mm/s)" % self.pulses_to_distance(pulseband[0]))
        elif pulses > pulseband[1]:
            raise self.gcode.error("target speed too high, no suitable power found. (max %.1f mm/s)" % self.pulses_to_distance(pulseband[1]))

        return [power for pulses, power in self.pulses_to_power.items() if pulses >= pulses][0]

    def distance_to_pulses(self, distance):
        return int(distance / self.resolution)

    def pulses_to_distance(self, pulses):
        return pulses * self.resolution

    def get_encoder_distance(self):
        return self.pulses_to_distance(self._pulses)

    def get_encoder_pulses(self):
        return self._pulses - self._initial_encoder_position

    def get_encoder_velocity(self):
        return self.pulses_to_distance(self._velocity[-1])

    def reset_encoder(self):
        self.log("Resetting encoder. Current pulses: %d" % self._pulses)
        self._initial_encoder_position = self._pulses

    @property
    def _backward(self):
        return 0 if self._forward else 1

    def stop(self):
        self.set_speed(None, 0.0)

    def check_busy(self, eventtime):
        result = self.control.check_busy(eventtime,
            self.smoothed_velocity, self.target_speed)
        self.log("check_busy: smoothed_velocity=%.3f target_speed=%.3f => %s" % (
            self.smoothed_velocity, self.target_speed, result))
        return result

    def _wait_for_speed(self):
        systime = self.reactor.monotonic()
        while not self.printer.is_shutdown() and self.check_busy(systime):
            self.toolhead.get_last_move_time()
            systime = self.reactor.pause(systime + 1.)

    def set_power(self, power):
        self.base_power = power
        self.gcrq.send_async_request((power, self.direction))

    def set_speed(self, direction, speed, wait=False):
        self._velocity = []

        if speed is None:
            power = None
        elif speed == 0.0:
            power = 0.0
            self.target_speed = 0.0
        elif speed != self.target_speed:
            power = self.speed_to_power(speed)
            self.target_speed = speed
        else:
            power = None

        self.set_power(power)

        if wait and speed:
            self._wait_for_speed()

    cmd_MMU_MOTOR_help = "Set motor speed and direction."
    def cmd_MMU_MOTOR(self, gcmd):
        self.mmu.log_to_file(gcmd.get_commandline())

        new_speed = None
        new_dir = None

        forward = gcmd.get_int("FORWARD", None, minval=0, maxval=1)
        if forward is not None:
            old_dir = FORWARD if self.direction == self._forward else BACKWARD
            self._forward = forward
            new_speed = 0
            new_dir = self._forward if old_dir == FORWARD else self._backward
            self.gcode.respond_info("Updated motor forward direction to %d" % self._forward)

        Kp = gcmd.get_float("KP", None, minval=0.)
        if Kp is not None:
            self.control.Kp = Kp / PID_PARAM_BASE
            self.gcode.respond_info("Updated motor control Kp to %.3f" % Kp)
        Ki = gcmd.get_float("KI", None, minval=0.)
        if Ki is not None:
            self.control.Ki = Ki / PID_PARAM_BASE
            self.gcode.respond_info("Updated motor control Ki to %.3f" % Ki)
        Kd = gcmd.get_float("KD", None, minval=0.)
        if Kd is not None:
            self.control.Kd = Kd / PID_PARAM_BASE
            self.gcode.respond_info("Updated motor control Kd to %.3f" % Kd)

        speed = gcmd.get_int("SPEED", None, minval=0, maxval=200)
        if speed is not None:
            new_speed = speed

        dir = gcmd.get("DIRECTION", None)
        if dir is not None:
            dir = dir.upper()
            if dir == "FORWARD":
                new_dir = self._forward
            elif dir == "BACKWARD":
                new_dir = self._backward
            else:
                raise gcmd.error("Invalid direction. Directions are: FORWARD, BACKWARD")

        self.set_speed(new_dir, new_speed)

        duration = gcmd.get_float("DURATION", None, minval=0.)
        if duration is not None:
            waketime = self.reactor.monotonic() + duration
            self.reactor.register_callback(lambda pt: self.stop(), waketime) # Schedule off

    cmd_MMU_MOTOR_CALIBRATE_help = "Run PID calibration test"
    def cmd_MMU_MOTOR_CALIBRATE(self, gcmd):
        target = gcmd.get_float('SPEED')
        self.toolhead.get_last_move_time()

        old_control = self.set_control(None)

        # 'warm up' motor / encoder
        self.set_power(100)
        self.toolhead.dwell(MOTOR_WARMUP_DELAY)
        self.set_power(0)
        self.toolhead.dwell(MOTOR_STOP_DELAY)

        last_pulses = 99999999
        self.pulses_to_power = {}
        for power in range(100, -1, -5):
            self.reset_encoder()
            self.set_power(power)
            self.toolhead.dwell(MOTOR_START_DELAY)

            start_time = self.reactor.monotonic()
            self.toolhead.dwell(MOTOR_RUN_DELAY)
            duration = self.reactor.monotonic() - start_time
            if duration < MOTOR_RUN_DELAY:
                self.toolhead.dwell(MOTOR_RUN_DELAY - duration)
                duration = self.reactor.monotonic() - start_time

            self.set_power(0)
            self.toolhead.dwell(MOTOR_STOP_DELAY)

            pulses = max(0, self.get_encoder_pulses() / duration)
            pulses = pulses if pulses > 100. else 0.

            if last_pulses <= pulses and power <= 100.0:
                raise gcmd.error("Pulses did not decrease with power: last %d, current %d" % (last_pulses, pulses))
            self.log("Power: %d =>  %.1f pulses/s (duration: %.3f sec)" % (power, pulses, duration))
            gcmd.respond_info("Power: %d =>  %.1f pulses/s (duration: %.3f sec)" % (power, pulses, duration))
            self.pulses_to_power[pulses] = power
            last_pulses = pulses

            if pulses < 100.:
                break

        self.mmu.save_variable(mmu.Mmu.VARS_MMU_MOTOR_POWER_MAP_PREFIX + self.name, self.pulses_to_power, write=True)
        pulseband = (
            min([i for i in self.pulses_to_power.keys() if i > 0]),
            max(self.pulses_to_power.keys())
        )
        power = self.speed_to_power(target)
        gcmd.respond_info("Saved motor power map.\n"
                            "Min power: %d => %d mm/s\n"
                            "Max power: %d => %d mm/s\n"
                            "Starting power for target speed %.1f mm/s (%.1f pulses/s): %d"
                            % (
                                self.pulses_to_power[pulseband[0]], self.pulses_to_distance(pulseband[0]),
                                self.pulses_to_power[pulseband[1]], self.pulses_to_distance(pulseband[1]),
                                target, self.distance_to_pulses(target), power
                            )
                        )

        self.set_power(power)
        self.toolhead.dwell(MOTOR_START_DELAY)

        self.set_control(old_control)
        calibrate = pid_calibrate.ControlAutoTune(self, target)
        old_control = self.set_control(calibrate)
        try:
            self.set_speed(FORWARD, target, True)
        except self.printer.command_error as e:
            self.set_control(old_control)
            raise
        self.set_control(old_control)
        if calibrate.check_busy(0., 0., 0.):
            raise gcmd.error("pid_calibrate interrupted")
        # Log and report results
        Kp, Ki, Kd = calibrate.calc_final_pid()
        logging.info("Autotune: final: Kp=%f Ki=%f Kd=%f", Kp, Ki, Kd)
        self.control.Kp = Kp / PID_PARAM_BASE
        self.control.Ki = Ki / PID_PARAM_BASE
        self.control.Kd = Kd / PID_PARAM_BASE
        gcmd.respond_info(
            "PID parameters:\n"
            "pid_Kp: %.3f\n"
            "pid_Ki: %.3f\n"
            "pid_Kd: %.3f\n"
            "Please update your config file with these parameters and restart the printer." % (Kp, Ki, Kd))
        gcmd.respond_info("Motor calibration complete.")

    def start_homing(self, endstops, speed, accel):
        self.endstops = endstops
        self.speed = speed
        self.accel = accel

    def homing_move(self, movepos, speed, probe_pos=False,
                    triggered=True, check_triggered=True):
        endstop_value = 1 if triggered else 0
        endstop = self.endstops[0] # Assume only one endstop for motor rail
        self.toolhead.dwell(HOMING_START_DELAY)

        # todo: rework

        distance = abs(movepos[1] - self.mmu_toolhead.get_position()[1])

        # distance needed to accelerate to velocity
        d_accel = self.speed**2 / (2 * self.accel)

        # minimum distance to reach max speed and decelerate
        d_min = 2 * d_accel

        if distance >= d_min:
            # --- trapezoidal profile ---
            t_accel = self.speed / self.accel
            d_cruise = distance - d_min
            t_cruise = d_cruise / self.speed
            timeout = 2 * t_accel + t_cruise
        else:
            # peak speed reached
            v_peak = math.sqrt(distance * self.accel)
            t_accel = v_peak / self.accel
            timeout = 2 * t_accel

        # wait for endstop
        print_time = self.toolhead.get_last_move_time() # TODO: mmu_toolhead instead of toolhead ??
        timeout += print_time
        while endstop[0].query_endstop(print_time) != endstop_value:
            self.toolhead.dwell(ENDSTOP_SAMPLE_TIME)
            print_time = self.toolhead.get_last_move_time()

            if check_triggered and print_time > timeout:
                raise self.printer.command_error("No trigger on %s after full movement" % (endstop[1],))

        return movepos

    def move(self, position, speed):
        move_interval = .5

        end_position = position[1]
        start_position = self.mmu_toolhead.get_position()[1]

        if end_position == start_position:
            return
        elif start_position < end_position:
            motor_direction = FORWARD
        else:
            motor_direction = BACKWARD

        self.reset_encoder()
        distance = abs(end_position - start_position)
        self.set_speed(motor_direction, speed)

        while distance > 0:
            self.mmu_toolhead.dwell(move_interval)

            encoder_position = self.get_encoder_distance()
            self.log("Move --> Distance: %.3f mm, Pulses: %d, Remaining %.3f mm, Velocity: %.1f mm/s" % (
                encoder_position,
                self.get_encoder_pulses(),
                distance - encoder_position,
                self.get_encoder_velocity()))

        self.set_position(position)




def load_config_prefix(config):
    return MMUMotor(config)
