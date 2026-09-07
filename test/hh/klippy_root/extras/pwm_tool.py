# Happy Hare MMU Software
#
# Copyright (C) 2022-2026  moggieuk#6538 (discord)
#                          moggieuk@hotmail.com
#
# Goal: Provide queued PWM behavior for the fake Klipper runtime.
#
#
# (\_/)
# ( *,*)
# (")_(") Happy Hare Ready
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#

# Happy Hare test harness. GNU GPLv3.
from mcu import MCU_pwm


class MCU_queued_pwm(MCU_pwm):
    """Expose the PWM pin API; BLDC queue tests validate scheduling separately."""

    def __init__(self, config, pin_params):
        mcu = config.get_printer().lookup_object(
            'mcu' if pin_params['chip_name'] == 'mcu'
            else 'mcu ' + pin_params['chip_name'])
        super().__init__(mcu, pin_params)
