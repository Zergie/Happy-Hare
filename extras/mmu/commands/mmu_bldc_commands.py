# Happy Hare MMU Software
#
# Copyright (C) 2022-2026  moggieuk#6538 (discord)
#                          moggieuk@hotmail.com
#
# Goal: Parse and orchestrate per-unit BLDC calibration commands.
#
#
# (\_/)
# ( *,*)
# (")_(") Happy Hare Ready
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#

from .mmu_base_command import BaseCommand, CATEGORY_TESTING
from ..mmu_utils import MmuError


class MmuCalibrateBldcCommand(BaseCommand):
    """Calibrate the selected unit; retain the v3 MOTOR index as an alias."""

    def __init__(self, mmu):
        super().__init__(mmu)
        self.register(
            name='MMU_CALIBRATE_BLDC', handler=self._run,
            help_brief='Calibrate the BLDC PWM/RPM map',
            help_params='UNIT=<name/index> or MOTOR=<index>, POINTS=10, PASSES=3, SAVE=1',
            category=CATEGORY_TESTING)

    def _run(self, gcmd):
        if self.check_if_disabled() or self.check_if_printing() or self.check_if_bypass():
            return
        mmu = self.mmu
        motor = gcmd.get_int('MOTOR', -1, minval=-1,
                             maxval=mmu.mmu_machine.num_units - 1)
        unit_name = gcmd.get('UNIT', None)
        if motor >= 0 and unit_name is not None:
            raise gcmd.error('Specify UNIT or MOTOR, not both')
        if motor >= 0:
            unit = mmu.mmu_machine.get_mmu_unit_by_index(motor)
        elif unit_name is not None:
            unit = self._lookup_mmu_unit(mmu.mmu_machine, unit_name)
        else:
            unit = mmu.mmu_unit()
        if unit is None or not hasattr(unit.drives[0], 'bldc'):
            raise gcmd.error('Selected unit has no BLDC drive')
        if unit is not mmu.mmu_unit() or not unit.owns_gate(mmu.gate_selected):
            raise gcmd.error('Select a gate on this unit before BLDC calibration')
        drive = unit.drives[0]
        bldc = drive.bldc
        points = gcmd.get_int('POINTS', bldc.CALIBRATION_DEFAULT_POINTS,
                              minval=bldc.CALIBRATION_MIN_POINTS, maxval=40)
        passes = gcmd.get_int('PASSES', 3, minval=1, maxval=10)
        save = gcmd.get_int('SAVE', 1, minval=0, maxval=1)
        with mmu.wrap_sync_gear_to_extruder(), mmu.wrap_suspend_filament_monitoring():
            try:
                mmu.calibrating = True
                payload = bldc.calibrate_pwm_rpm_map(points, passes, mmu.log_always)
                mmu.log_always('BLDC calibration for %s:\n%s' % (
                    unit.name, '\n'.join('PWM %.3f: %.1f RPM' % (p['pwm'], p['rpm'])
                                         for p in payload['points'])))
                if save:
                    drive.save_calibration()
            except MmuError as error:
                raise gcmd.error(str(error))
            finally:
                bldc.stop()
                mmu.calibrating = False
