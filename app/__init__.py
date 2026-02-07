#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
    BEATBRIDGE
    Main BPM Daemon Module (src/__init__.py)
"""
__author__ = "mail@michael.welte.de"
__copyright__ = "Copyright © 2025-2026 by Michael Welte. All rights reserved."


# Load configuration reference
from config import Config as cfg

# Imports
from logging.handlers import RotatingFileHandler
from pathlib import Path
from pythonjsonlogger import jsonlogger

# Setup logging with custom logger class instance
from utils.bpm_custom_logger import CustomLogger
custom_logger = CustomLogger(cfg.APP_MODULE)
custom_logger.setLevel(cfg.LOG_LEVEL)

# Make sure that there is a logging folder
log_file_dir = Path(cfg.APP_LOG_FILE_NAME).parent
log_file_dir.mkdir(parents=True, exist_ok=True)

# Initialize logging on rotation
log_file_handler = RotatingFileHandler( filename = cfg.APP_LOG_FILE_NAME,
                                        encoding = cfg.APP_LOG_FILE_ENCODING,
                                        backupCount = cfg.APP_LOG_FILE_ROTATING_BAKUPS,                                        
                                        maxBytes = cfg.APP_LOG_FILE_SIZE_MAXIMUM )
log_file_handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(message)s"))
custom_logger.addHandler(log_file_handler)
