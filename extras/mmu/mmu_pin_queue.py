# Happy Hare MMU Software
#
# Copyright (C) 2022-2026  moggieuk#6538 (discord)
#                          moggieuk@hotmail.com
#
# Goal: Share queued pin dispatch between MMU motors on the same MCU.
#
#
# (\_/)
# ( *,*)
# (")_(") Happy Hare Ready
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#

class MmuPinRequestQueue:
    """One queued pin dispatcher shared by all MMU motors on an MCU."""

    REGISTRY_ATTRIBUTE = '_hh_pin_request_queues'

    @classmethod
    def get_for_mcu(cls, config, mcu):
        printer = config.get_printer()
        queues = getattr(printer, cls.REGISTRY_ATTRIBUTE, None)
        if queues is None:
            queues = {}
            setattr(printer, cls.REGISTRY_ATTRIBUTE, queues)
        if mcu not in queues:
            queues[mcu] = cls(config, mcu)
        return queues[mcu]

    def __init__(self, config, mcu):
        self.native_mode = False
        try:
            from .. import output_pin
        except ImportError:
            output_pin = None
        if output_pin is not None and hasattr(output_pin, 'GCodeRequestQueue'):
            self.native_mode = True
            self.queue = output_pin.GCodeRequestQueue(
                config, mcu, self._dispatch
            )
            return

        from .unit.mmu_espooler import GCodeRequestQueue as FallbackGCodeRequestQueue
        self.queue = FallbackGCodeRequestQueue(config, mcu, self._dispatch)

    def _dispatch(self, print_time, request):
        callback, value = request
        return callback(print_time, value)

    def send(self, request, print_time):
        self.queue.send_async_request(request, print_time)

