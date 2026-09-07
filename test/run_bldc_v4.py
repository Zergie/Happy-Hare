# Happy Hare MMU Software
#
# Copyright (C) 2022-2026  moggieuk#6538 (discord)
#                          moggieuk@hotmail.com
#
# Goal: Run BLDC controller and integration regressions on supported hosts.
#
#
# (\_/)
# ( *,*)
# (")_(") Happy Hare Ready
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#

"""Run BLDC integration regressions, including on Windows with Git Bash.

Usage: python -X utf8 -m test.run_bldc_v4 [unittest module names]
Includes controller tests through the standard unittest loader.
"""
import os
from pathlib import Path
import subprocess
import sys
import unittest


def configure_windows():
    if os.name != 'nt':
        return
    import jinja2

    bash = os.environ.get('HH_TEST_BASH') or str(
        Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/bin/bash.exe')
    if not Path(bash).is_file():
        raise RuntimeError('Install Git Bash or set HH_TEST_BASH to bash.exe')
    original_popen = subprocess.Popen

    class BashPopen(original_popen):
        def __init__(self, args, *pos, **kwargs):
            if kwargs.get('shell'):
                args = [bash, '-c', 'export PATH=/usr/bin:/bin:$PATH\n' + args]
                kwargs['shell'] = False
            super().__init__(args, *pos, **kwargs)

    subprocess.Popen = BashPopen
    original_get_source = jinja2.FileSystemLoader.get_source

    def get_source(loader, environment, template):
        return original_get_source(loader, environment, template.replace('\\', '/'))

    jinja2.FileSystemLoader.get_source = get_source


def main():
    configure_windows()
    names = sys.argv[1:] or [
        'test.test_mmu_gear_bldc', 'test.test_mmu_bldc_v4', 'test.test_mmu_bootup',
        'test.test_mmu_compound_endstop', 'test.test_mmu_current_nesting',
    ]
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return int(not result.wasSuccessful())


if __name__ == '__main__':
    raise SystemExit(main())
