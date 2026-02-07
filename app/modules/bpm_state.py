#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
    BEATBRIDGE
    Global State Module (bpm_state.py)
"""
__author__ = "mail@michael.welte.de"
__copyright__ = "Copyright © 2025-2026 by Michael Welte. All rights reserved."


import threading
import time

class BPMState:
    def __init__(self, bpm_init=120.0):
        self.lock = threading.Lock()
        self.bpm = bpm_init
        self.source = "HOLD"   # AUDIO | TAP | HOLD
        self.last_update = time.monotonic()

    def set_bpm(self, bpm, source):
        with self.lock:
            self.bpm = bpm
            self.source = source
            self.last_update = time.monotonic()

    def get(self):
        with self.lock:
            return self.bpm, self.source
