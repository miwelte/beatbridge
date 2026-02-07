#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
    BEATBRIDGE
    Main entry point for the BPM Daemon
"""
__author__ = "mail@michael.welte.de"
__copyright__ = "Copyright © 2025-2026 by Michael Welte. All rights reserved."


from config import Config as cfg
from __init__ import custom_logger as clog

from modules.bpm_state import BPMState



state = BPMState()
beat_queue = Queue()


# Main
def main():
    clog.setLevel(cfg.LOG_LEVEL)
    clog.info(f"Starting '{cfg.APP_NAME}' on '{cfg.RUN_ENV_INFO}' in '{cfg.APP_ENV}' mode with log-level set to '{clog.getLevel()}'.")

    start(audio_bpm_thread)
    start(tap_thread)
    start(clock_thread)
    start(midi_thread)

    wait_forever()

if __name__ == "__main__":
    main()