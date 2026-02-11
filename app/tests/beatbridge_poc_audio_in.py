#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
    BEATBRIDGE
    PoC: Audio Input (beatbridge_poc_audio_in.py)
"""
__author__ = "mail@michael.welte.de"
__copyright__ = "Copyright © 2025-2026 by Michael Welte. All rights reserved."


import time
import numpy as np
import sounddevice as sd


SAMPLE_RATE = 48_000
CHANNELS = 2
BLOCK_SIZE = 1024  # ~21 ms @ 48 kHz
DEVICE_INDEX = 0   # HiFiBerry DAC+ADC (hw:2,0)

RMS_SIGNAL_PRESENT = 0.002
RMS_PEAK = 0.05

PEAK_THRESHOLD_RED = 0.7
PEAK_THRESHOLD_YLW = 0.5
PEAK_THRESHOLD_GRN = 0.0

PEAK_COOLDOWN_S = 0.10

last_peak_ts = 0.0


def rms_level(block: np.ndarray) -> float:
    # block shape: (frames, channels), float32 in [-1..1]
    mono = block.mean(axis=1)
    return float(np.sqrt(np.mean(mono * mono)))


def main() -> None:
    global last_peak_ts
    print("Audio input PoC (RMS + Peak). Ctrl+C to stop.")

    with sd.InputStream(
        device=DEVICE_INDEX,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="float32",
    ) as stream:
        while True:
            block, overflowed = stream.read(BLOCK_SIZE)
            if overflowed:
                print("WARNING: overflow (try bigger blocksize)")

            rms = rms_level(block)
            peak = float(np.max(np.abs(block)))
            now = time.time()

            if rms >= RMS_SIGNAL_PRESENT:
                msg = f"RMS={rms:.5f} PEAK={peak:.3f}"
                if rms >= RMS_PEAK and (now - last_peak_ts) >= PEAK_COOLDOWN_S:
                    last_peak_ts = now
                    msg += "  -> PEAK"
                if peak > PEAK_THRESHOLD_RED:
                    msg += " [RED]"
                elif peak > PEAK_THRESHOLD_YLW:
                    msg += " [YLW]"
                elif peak > PEAK_THRESHOLD_GRN:
                    msg += " [GRN]"
                print(msg)
            else:
                print(f"RMS={rms:.5f} (low)")

            time.sleep(0.01)


if __name__ == "__main__":
    main()
