#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
    BEATBRIDGE
    Main entry point for the BPM Daemon
"""
__author__ = "mail@michael.welte.de"
__copyright__ = "Copyright © 2025-2026 by Michael Welte. All rights reserved."


state = BPMState()
beat_queue = Queue()

start(audio_bpm_thread)
start(tap_thread)
start(clock_thread)
start(midi_thread)

wait_forever()
