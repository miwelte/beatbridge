#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
    BEATBRIDGE
    Tap Input Module (bpm_tap_in.py)
"""
__author__ = "mail@michael.welte.de"
__copyright__ = "Copyright © 2025-2026 by Michael Welte. All rights reserved."

# Common imports
import time
from statistics import median

# App imports
from bpm_state import BPMState


def tap_thread(state: BPMState):
    taps = []

    while True:
        wait_for_tap_event()  # GPIO oder MIDI
        now = time.monotonic()
        taps.append(now)

        if len(taps) >= 4:
            intervals = [taps[i+1]-taps[i] for i in range(len(taps)-1)]
            interval = median(intervals[-3:])
            bpm = 60.0 / interval
            state.set_bpm(bpm, source="TAP")
            taps = taps[-2:]  # reset window
