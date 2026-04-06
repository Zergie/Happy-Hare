# Happy Hare MMU Software
#
# (\_/)
# ( *,*)
# (")_(") Happy Hare Ready
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#
import logging
from . import pulse_counter

MAX_SCHEDULE_TIME = 5.0

ACCEL_THRESHOLD = 10.  # mm/s^2
IDLE_THRESHOLD = 1.  # mm/s

STARTUP_DELAY = 0.5
STOPPING_DELAY = 1.0

# motor states
STATE_IDLE = 'IDLE'
STATE_CRUISING = 'CRUISING'
STATE_ACCELERATING = 'ACCELERATING'
STATE_DECELERATING = 'DECELERATING'

class MMUMotorEncoder():
    def __init__(self, config):
        self.name = config.get_name().split()[-1]
        self.printer = config.get_printer()
        self.smooth_time = config.getfloat('smooth_time', 1., above=0.)
        self.inv_smooth_time = 1. / self.smooth_time
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
        self.state = STATE_IDLE

    def get_state(self):
        return {
            'state': self.state,
            'pulses': self._pulses - self._initial_encoder_position,
            'distance': self.get_position(),
            'velocity': self.velocity,
            'smoothed_velocity': self.smoothed_velocity,
        }

    def distance_to_pulses(self, distance):
        return int(distance / self.resolution)

    def pulses_to_distance(self, pulses):
        return pulses * self.resolution

    def get_position(self):
        return self.pulses_to_distance(self._pulses - self._initial_encoder_position)

    def get_velocity(self):
        return self.pulses_to_distance(self.smoothed_velocity)

    def reset_position(self):
        self._initial_encoder_position = self._pulses

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
        else:
            velocity = self.pulses_to_distance(new_pulses / time_diff)
            self._pulses += new_pulses

        accel = self.pulses_to_distance((velocity - self.velocity) / (time_diff if time_diff > 0. else 0.0))
        if accel >= ACCEL_THRESHOLD:
            self.state = STATE_ACCELERATING
        elif accel <= -ACCEL_THRESHOLD:
            self.state = STATE_DECELERATING
        elif velocity < IDLE_THRESHOLD:
            self.state = STATE_IDLE
        else:
            self.state = STATE_CRUISING

        velocity_diff = velocity - self.smoothed_velocity
        adj_time = min(time_diff * self.inv_smooth_time, 1.)
        self.smoothed_velocity += velocity_diff * adj_time
        self.velocity = velocity
        self._last_pulses = pulses

def load_config_prefix(config):
    return MMUMotorEncoder(config)
