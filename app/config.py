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

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RUN_ENV = os.getenv("RUN_ENV", "localhost")


class Config:

   # Project settings 
    APP_ENV = env("ENVIRONMENT") or None
    APP_ENV_IS_DEV = APP_ENV == "development"
    APP_ENV_IS_PROD = APP_ENV == "production"
    RUN_ENV_IS_LOCALHOST = RUN_ENV == "localhost"
    RUN_ENV_IS_DOCKER = RUN_ENV == "docker"
    RUN_ENV_IS_K8S = RUN_ENV == "k8s"
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

