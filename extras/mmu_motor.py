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
from . import pulse_counter, output_pin, heaters, homing
from . import mmu_espooler, mmu


MAX_SCHEDULE_TIME = 5.0

ACCEL_THRESHOLD = 10.  # mm/s^2
IDLE_THRESHOLD = 1.  # mm/s

STARTUP_DELAY = 0.5
STOPPING_DELAY = 1.0

# directions
FORWARD  = 1
BACKWARD = 0

# motor state
STATE_IDLE = 'IDLE'
STATE_CRUISING = 'CRUISING'
STATE_ACCELERATING = 'ACCELERATING'
STATE_DECELERATING = 'DECELERATING'

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
        self.smooth_time = config.getfloat('smooth_time', 1., above=0.)
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
        self._pulse_time = None
        self._pulses = self._last_pulses = self._initial_encoder_position = 0
        self.velocity = 0.
        self.smoothed_velocity = 0.

        # PID control
        self.Ki = config.getfloat('pid_Ki', 160.0) / heaters.PID_PARAM_BASE
        self.trim_power = 0.

        # motor state
        self.motor_start = 0.
        self.state = STATE_IDLE

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

        # G-code commands
        self.gcode = self.printer.lookup_object('gcode')
        self.gcode.register_mux_command("MMU_MOTOR", "MOTOR", self.name,
                                        self.cmd_MMU_MOTOR,
                                        desc=self.cmd_MMU_MOTOR_help)
        self.gcode.register_mux_command("MMU_MOTOR_CALIBRATE", "MOTOR", self.name,
                                        self.cmd_MMU_MOTOR_CALIBRATE,
                                        desc=self.cmd_MMU_MOTOR_CALIBRATE_help)
        self.gcode.register_mux_command("MMU_MOTOR_TEST", "MOTOR", self.name,
                                        self.cmd_MMU_MOTOR_TEST,
                                        desc=self.cmd_MMU_MOTOR_TEST_help) #TODO: remove

        # Register for connect event
        self.printer.register_event_handler('klippy:connect', self.handle_connect)

    def handle_connect(self):
        self.mmu = self.printer.lookup_object('mmu')
        self.toolhead = self.printer.lookup_object('toolhead')
        self.save_variables = self.printer.lookup_object('save_variables')
        self.mmu_toolhead = self.mmu.mmu_toolhead
        self.pulses_to_power = self.save_variables.allVariables.get(mmu.Mmu.VARS_MMU_MOTOR_POWER_MAP_PREFIX + self.name, {})

    # Callback for MCU_counter
    def _counter_callback(self, print_time, pulses, pulse_time):
        if self._pulse_time is None:  # First sample
            self._pulse_time = pulse_time
            self._last_pulses = pulses
            return

        if pulse_time > self._pulse_time:
            time_diff = pulse_time - self._pulse_time
            self._pulse_time = pulse_time
            new_pulses = pulses - self._last_pulses
        else:  # No counts since last sample
            time_diff = self.sample_time
            self._pulse_time = max(print_time, self._pulse_time + self.sample_time + .0005)
            new_pulses = 0

        if new_pulses == 0:
            velocity = 0.0

            if self.power == 0.0 and -0.1 < time_diff < 0.0:
                self.log("-----> Motor stopped. Total pulses: %d, self.pulses: %d, time_diff=%.6f s" % (
                    self.get_encoder_pulses(),
                    time_diff))
        else:
            velocity = self.pulses_to_distance(new_pulses / time_diff)
            self._pulses += new_pulses

        accel = self.pulses_to_distance((velocity - self.velocity) / (time_diff if time_diff > 0. else 0.0))
        if accel >= ACCEL_THRESHOLD:
            state = STATE_ACCELERATING
        elif accel <= -ACCEL_THRESHOLD:
            state = STATE_DECELERATING
        elif velocity < IDLE_THRESHOLD:
            state = STATE_IDLE
        else:
            state = STATE_CRUISING

        velocity_diff = velocity - self.smoothed_velocity
        adj_time = min(time_diff * self.inv_smooth_time, 1.)
        self.smoothed_velocity += velocity_diff * adj_time

        # PID speed control
        if self.target_speed <= 1.0:
            pass
        elif self.motor_start + STARTUP_DELAY > pulse_time:
            self.log("-----> PID control: warming up, time remaining %.3f s" % (self.motor_start + STARTUP_DELAY - pulse_time,))
        else:
            error = self.target_speed - self.pulses_to_distance(velocity)
            if abs(error) < .1:
                error = 0.

            # compute tentative output (before integrating):
            power_unsat = self.base_power + self.trim_power

            # anti-windup: only integrate if it helps
            at_max = (power_unsat >= 100. and error > 0)
            at_min = (power_unsat <= 0. and error < 0)

            if not (at_max or at_min):
                self.trim_power += self.Ki * error * time_diff

            # clamp I_trim
            self.trim_power = max(-50., min(self.trim_power, 100.))

            # final output clamp
            power = max(0., min(self.base_power + self.trim_power, 100.))

            self.gcode.respond_info("PID control: vel=%.1f mm/s, error=%.1f mm/s, trim=%.1f => power=%.1f" % (
                self.pulses_to_distance(velocity),
                error,
                self.trim_power,
                power
            ))
            self.gcrq.send_async_request((power, None))

        self.state = state
        self.velocity = velocity
        self._last_pulses = pulses

    # This is the actual callback method to update pin signal (pwm or digital)
    def _set_pin(self, print_time, action):
        power, dir_value = action

        if power is not None:
            if self.power == 0.0 and power > 0.0:
                self.log("<----- Motor starting.")

            self.pwm_pin.set_pwm(print_time, power / 100.0)
            self.power = power

        if dir_value is not None:
            if self.direction != dir_value:
                self.dir_pin.set_digital(print_time, dir_value)
                self.direction = dir_value

    def get_pwm_delay(self):
        if self._pulse_time is None:
            delay = 999999999999.
        else:
            systime = self.reactor.monotonic()
            print_time = self.mcu.estimated_print_time(systime)
            delay = print_time - self._pulse_time + self.sample_time
        delay = max(self.sample_time * 2., delay) * 10.
        return delay

    def _wait_for_speed(self):
        systime = self.reactor.monotonic()
        while not self.printer.is_shutdown() and abs(self.smoothed_velocity - self.target_speed) > 1.0:
            self.toolhead.get_last_move_time()
            systime = self.reactor.pause(systime + max(.1, self.sample_time))

    def get_status(self, eventtime):
        return {
            'forward': self._forward,
            'encoder_pulses': min(0, self._pulses),
            'encoder_distance': min(0, self.get_encoder_distance()),
            'target_speed': self.target_speed,
            'power': self.power,
            'direction': self.direction,
            'smoothed_velocity': self.smoothed_velocity,
            'velocity_to_power': self.pulses_to_power,
            'Ki': self.Ki * heaters.PID_PARAM_BASE,
        }

    def log(self, msg):
        if self.name == "unit0":
            logging.info("mmu_motor %s: %s" % (self.name, msg))

    def speed_to_power(self, speed):
        pulses = self.distance_to_pulses(speed)
        pulseband = (
            min([0,] + [i for i in self.pulses_to_power.keys() if i > 0]),
            max(self.pulses_to_power.keys())
        )

        if pulses < pulseband[0]:
            raise self.gcode.error("target speed too low, no suitable power found. (min %.1f mm/s)" % self.pulses_to_distance(pulseband[0]))
        elif pulses > pulseband[1]:
            raise self.gcode.error("target speed too high, no suitable power found. (max %.1f mm/s)" % self.pulses_to_distance(pulseband[1]))

        powerband = sorted([power for p, power in self.pulses_to_power.items() if p >= pulses])
        # return powerband[1] if len(powerband) > 1 else powerband[0]
        return powerband[0]

    def distance_to_pulses(self, distance):
        return int(distance / self.resolution)

    def pulses_to_distance(self, pulses):
        return pulses * self.resolution

    def get_encoder_distance(self):
        return self.pulses_to_distance(self._pulses)

    def get_encoder_pulses(self):
        return self._pulses - self._initial_encoder_position

    def reset_encoder(self):
        self.log("Resetting encoder. Current pulses: %d" % self._pulses)
        self._initial_encoder_position = self._pulses

    @property
    def _backward(self):
        return 0 if self._forward else 1

    def stop(self):
        self.set_speed(0.0)

    def set_power(self, power, direction=None):
        if power is not None:
            self.motor_start = self.mcu.estimated_print_time(self.reactor.monotonic())
            self.base_power = power
            self.trim_power = 0.
        self.gcrq.send_async_request((power, self.direction if direction is None else direction))

    def set_speed(self, speed, direction=None, wait=False):
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

        if direction is None:
            dir = None
        elif direction == FORWARD:
            dir = self._forward
        elif direction == BACKWARD:
            dir = self._backward

        self.log("set_speed: speed=%.3f => power=%.1f" % (speed, power if power is not None else -1.))
        self.set_power(power, dir)

        if wait and speed:
            self._wait_for_speed()

    def wait_for_state(self, desired_state, timeout):
        start_time = self.reactor.monotonic()
        while not self.printer.is_shutdown() and self.state != desired_state:
            self.toolhead.get_last_move_time()
            systime = self.reactor.monotonic()
            if (systime - start_time) >= timeout:
                return False
            self.reactor.pause(systime + max(.1, self.sample_time))
        return True


    cmd_MMU_MOTOR_TEST_help = ""
    def cmd_MMU_MOTOR_TEST(self, gcmd):
        self.gcode.respond_info("Motor test complete. Encoder got %d pulses." % self.get_encoder_pulses())

    cmd_MMU_MOTOR_help = "Set motor speed and direction."
    def cmd_MMU_MOTOR(self, gcmd):
        self.mmu.log_to_file(gcmd.get_commandline())

        new_speed = self.speed
        new_dir = self.direction

        forward = gcmd.get_int("FORWARD", None, minval=0, maxval=1)
        if forward is not None:
            old_dir = FORWARD if self.direction == self._forward else BACKWARD
            self._forward = forward
            new_speed = 0
            new_dir = self._forward if old_dir == FORWARD else self._backward
            self.gcode.respond_info("Updated motor forward direction to %d" % self._forward)

        Ki = gcmd.get_float("KI", None, minval=0.)
        if Ki is not None:
            self.Ki = Ki / heaters.PID_PARAM_BASE
            self.trim_power = 0.
            self.gcode.respond_info("Updated motor control Ki to %.3f" % Ki)

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

        self.set_speed(new_speed, new_dir)

        duration = gcmd.get_float("DURATION", None, minval=0.)
        if duration is not None:
            waketime = self.reactor.monotonic() + duration
            self.reactor.register_callback(lambda pt: self.stop(), waketime) # Schedule off

    cmd_MMU_MOTOR_CALIBRATE_help = "Run PID calibration test"
    def cmd_MMU_MOTOR_CALIBRATE(self, gcmd):
        target = gcmd.get_float('SPEED')
        self.toolhead.get_last_move_time()

        old_ki = self.Ki
        self.Ki = 0.0  # disable PID during calibration

        last_velocity = 99999999
        smooth_time = self.smooth_time
        self.pulses_to_power = {}
        for power in range(100, -1, -5):
            velocity_samples = []
            while len(velocity_samples) < 2 or abs(velocity_samples[-1] - velocity_samples[-2]) > 1.: # measument should be +-1 pulses/s
            # if True:
                self.set_power(power)
                if not self.wait_for_state(STATE_CRUISING, STARTUP_DELAY):
                    gcmd.respond_info("Warning: Motor did not reach cruising state at power %d" % power)

                duration = 0.
                start_time = self.reactor.monotonic()
                while duration < smooth_time:
                    self.toolhead.dwell(smooth_time - duration)
                    duration = self.reactor.monotonic() - start_time

                velocity = self.smoothed_velocity
                if velocity < 40.0:
                    velocity = 0.0
                velocity_samples.append(velocity)

                self.log("Power: %d =>  %.1f pulses/s (duration: %.3f sec)" % (power, velocity, duration))
                gcmd.respond_info("Power: %d => %.1f mm/s (%.1f pulses/s, duration: %.3f s)" % (power, self.pulses_to_distance(velocity), velocity, duration))

            self.set_power(0)
            if not self.wait_for_state(STATE_IDLE, STOPPING_DELAY):
                gcmd.respond_info("Warning: Motor did not stop properly at power %d" % power)

            velocity = (velocity_samples[-1] + velocity_samples[-2]) / 2.
            gcmd.respond_info("Average velocity at power %d: %.1f mm/s (%.1f pulses/s)" % (power, self.pulses_to_distance(velocity), velocity))
            if velocity - last_velocity > 10.:
                raise gcmd.error("Speed did not decrease with power: last %.1f mm/s, current %.1f mm/s" % (self.pulses_to_distance(last_velocity), self.pulses_to_distance(velocity)))
            self.pulses_to_power[velocity] = power
            last_velocity = velocity

            if velocity < 1.:
                break

        self.Ki = old_ki  # restore PID
        self.mmu.save_variable(mmu.Mmu.VARS_MMU_MOTOR_POWER_MAP_PREFIX + self.name, self.pulses_to_power, write=True)
        pulseband = (
            min([0,] + [i for i in self.pulses_to_power.keys() if i > 0]),
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
        gcmd.respond_info("Motor calibration complete.")

    def start_homing(self, endstops, speed, accel):
        self.endstops = endstops
        self.speed = speed
        self.accel = accel

    def homing_move(self, movepos, speed, probe_pos=False,
                    triggered=True, check_triggered=True):
        endstop_value = 1 if triggered else 0
        endstop = self.endstops[0] # Assume only one endstop for motor rail
        self.toolhead.dwell(homing.HOMING_START_DELAY)

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
            self.toolhead.dwell(homing.ENDSTOP_SAMPLE_TIME)
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
        self.set_speed(speed, motor_direction)

        while distance > 0:
            self.mmu_toolhead.dwell(move_interval)

            encoder_position = self.get_encoder_distance()
            self.log("Move --> Distance: %.3f mm, Pulses: %d, Remaining %.3f mm, Velocity: %.1f mm/s" % (
                encoder_position,
                self.get_encoder_pulses(),
                distance - encoder_position,
                self.pulses_to_distance(self.smoothed_velocity)))

        self.set_position(position)




def load_config_prefix(config):
    return MMUMotor(config)
