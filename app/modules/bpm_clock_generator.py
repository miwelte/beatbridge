#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
    BEATBRIDGE
    Clock Generator Module (clock_generator.py)
"""
__author__ = "mail@michael.welte.de"
__copyright__ = "Copyright © 2025-2026 by Michael Welte. All rights reserved."


import time
from sleep_until import sleep_until

from bpm_state import BPMState


def clock_thread(state: BPMState, beat_queue):
    next_tick = time.monotonic()

    while True:
        bpm, _ = state.get()
        interval = 60.0 / bpm

        next_tick += interval
        sleep_until(next_tick)
        beat_queue.put("BEAT")
