#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
    BEATBRIDGE
    Audio Analysis Module (bpm_audio_analysis.py)
"""
__author__ = "mail@michael.welte.de"
__copyright__ = "Copyright © 2025-2026 by Michael Welte. All rights reserved."

def audio_bpm_thread(state: BPMState):
    detector = AubioTempoDetector()
    EMA_ALPHA = 0.15
    bpm_ema = state.get()[0]

    while True:
        bpm_candidate, confidence = detector.process()
        if confidence >= 0.75:
            bpm_ema = EMA_ALPHA * bpm_candidate + (1 - EMA_ALPHA) * bpm_ema
            state.set_bpm(bpm_ema, source="AUDIO")
