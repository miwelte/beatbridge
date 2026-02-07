#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
    BEATBRIDGE
    Musical Instrument Digital Interface Output Module (midi_out.py)
"""
__author__ = "mail@michael.welte.de"
__copyright__ = "Copyright © 2025-2026 by Michael Welte. All rights reserved."


def midi_thread(beat_queue):
    midi = MidiOut(port="LAS")

    while True:
        beat_queue.get()
        midi.note_on(note=60, velocity=100)  # C3
        midi.note_off(note=60)
