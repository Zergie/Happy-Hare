# Happy Hare MMU Software
#
# Copyright (C) 2017-2025  Kevin O'Connor <kevin@koconnor.net>
#
# Goal: Pin Klipper request-queue behavior for standalone BLDC timing tests.
#
# Extracted unchanged from klippy/extras/output_pin.py at Klipper commit
# b0e6ca45fc7b239145375ee80eb9666552c12821. Keep the upstream queue logic
# intact when updating this fixture; it exercises native queue spacing and
# callback return handling without requiring a local Klipper installation.
#
# (\_/)
# ( *,*)
# (")_(") Happy Hare Ready
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#

# Helper code to queue g-code requests
class GCodeRequestQueue:
    def __init__(self, config, mcu, callback):
        self.printer = printer = config.get_printer()
        self.mcu = mcu
        self.callback = callback
        self.rqueue = []
        self.next_min_flush_time = 0.
        self.toolhead = None
        self.motion_queuing = printer.load_object(config, 'motion_queuing')
        self.motion_queuing.register_flush_callback(self._flush_notification)
        printer.register_event_handler("klippy:connect", self._handle_connect)
    def _handle_connect(self):
        self.toolhead = self.printer.lookup_object('toolhead')
    def _flush_notification(self, must_flush_time, max_step_gen_time):
        min_sched_time = self.mcu.min_schedule_time()
        rqueue = self.rqueue
        while rqueue:
            next_time = max(rqueue[0][0], self.next_min_flush_time)
            if next_time > must_flush_time:
                return
            # Skip requests that have been overridden with a following request
            pos = 0
            while pos + 1 < len(rqueue) and rqueue[pos + 1][0] <= next_time:
                pos += 1
            req_pt, req_val = rqueue[pos]
            # Invoke callback for the request
            ret = self.callback(next_time, req_val)
            if ret is not None:
                # Handle special cases
                action, next_min_time = ret
                self.next_min_flush_time = max(self.next_min_flush_time,
                                               next_min_time)
                if action == "discard":
                    del rqueue[:pos+1]
                    continue
                if action == "reschedule":
                    del rqueue[:pos]
                    continue
                if action == "repeat":
                    pos -= 1
            del rqueue[:pos+1]
            self.next_min_flush_time = max(self.next_min_flush_time,
                                           next_time + min_sched_time)
            # Ensure following queue items are flushed
            self.motion_queuing.note_mcu_movequeue_activity(
                self.next_min_flush_time, is_step_gen=False)
    def _queue_request(self, print_time, value):
        self.rqueue.append((print_time, value))
        self.motion_queuing.note_mcu_movequeue_activity(
            print_time, is_step_gen=False)
    def queue_gcode_request(self, value):
        self.toolhead.register_lookahead_callback(
            (lambda pt: self._queue_request(pt, value)))
    def send_async_request(self, value, print_time=None):
        min_sched_time = self.mcu.min_schedule_time()
        if print_time is None:
            systime = self.printer.get_reactor().monotonic()
            print_time = self.mcu.estimated_print_time(systime + min_sched_time)
        while 1:
            next_time = max(print_time, self.next_min_flush_time)
            # Invoke callback for the request
            action, next_min_time = "normal", 0.
            ret = self.callback(next_time, value)
            if ret is not None:
                # Handle special cases
                action, next_min_time = ret
                self.next_min_flush_time = max(self.next_min_flush_time,
                                               next_min_time)
                if action == "discard":
                    break
                if action == "reschedule":
                    continue
            self.next_min_flush_time = max(self.next_min_flush_time,
                                           next_time + min_sched_time)
            if action != "repeat":
                break
