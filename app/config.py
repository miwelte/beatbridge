#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
    BEATBRIDGE
    Configuration Module (config.py)
"""
__author__ = "mail@michael.welte.de"
__copyright__ = "Copyright © 2025-2026 by Michael Welte. All rights reserved."


import os
import logging
import pytz
from dotenv import load_dotenv
from urllib.parse import urljoin

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_FILE = ".env-beatbridge.dev"
RUN_ENV = os.getenv("RUN_ENV", "localhost")

# Use environment variables as defined in container compose declarations.
print("INFO: Determining 'BASE_DIR' and loading 'env' variables from container compose declarations.")
env = os.environ.get
if RUN_ENV == "localhost":
    if load_dotenv(os.path.join(BASE_DIR, ENV_FILE)):
        env = os.environ.get
        print(f"INFO: Dotenv loaded environment variables from localhost '{ENV_FILE}'.")
    else:
        raise Exception(f"ERROR: DotenvFile '{ENV_FILE}' not found.")


class Config:

   # Project settings 
    APP_ENV = env("ENVIRONMENT") or None
    APP_ENV_IS_DEV = APP_ENV == "development"
    APP_ENV_IS_PROD = APP_ENV == "production"
    RUN_ENV_INFO = str(RUN_ENV)
    LOG_LEVEL = logging.DEBUG if APP_ENV_IS_DEV else logging.INFO
    
    # Default date and time formats (do not change)
    DATE_TIME_FORMAT = "YYYY-MM-DD HH:mm:ss"
    TIME_ZONE = pytz.timezone('Europe/Berlin')
        
    # Application settings
    APP_COMPANY = "ITcares"
    APP_MODULE = "atsk"
    APP_NAME = "ITcares API NX MicroService: ATSK (Autotask PSA Cloud Application)"
    APP_PORT = 3002
    APP_VERSION = "1.0.12"
    APP_STATUS_INTERVAL = 2

    # Application Data
    APP_DATA_EXPORT_PATH = os.path.join(BASE_DIR, env("DATA_EXPORT_PATH"))

    # Application Logging
    APP_LOG_FILE_NAME = os.path.join(BASE_DIR, "logs", f"{APP_MODULE}_log.json")
    APP_LOG_FILE_ENCODING = "utf-8"
    APP_LOG_FILE_ROTATING_BAKUPS = 9
    APP_LOG_FILE_SIZE_MAXIMUM = (1024 * 1024)

