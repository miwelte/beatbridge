#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
    BEATBRIDGE
    Custom Logger Module (bpm_custom_logger.py)
"""
__author__ = "mail@michael.welte.de"
__copyright__ = "Copyright © 2025-2026 by Michael Welte. All rights reserved."


# Imports
import inspect
import logging
import os
from enum import Enum
from types import FrameType

from rich.console import Console
rc = Console(style="grey50")


# Setup logging with custom logger class instance
class CustomLogger(logging.Logger):
    """ 
        Custom logger class, derivated from python's jsonlogger to generate an extended message 
        with additional information about the calling class and function and then sending it 
        to the logger as well as returning the new composed message text.
    """
    class LogLevelEnum(str, Enum):
        CRITICAL = logging.getLevelName(logging.CRITICAL)
        ERROR    = logging.getLevelName(logging.ERROR)
        WARNING  = logging.getLevelName(logging.WARNING)
        INFO     = logging.getLevelName(logging.INFO)
        DEBUG    = logging.getLevelName(logging.DEBUG)

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def _logmsg_(self, frame, level: int, message, *args, **kwargs) -> str:
        """ 
            Build extended logging message.
        """        
        calling_class = frame.f_locals.get('self', None).__class__.__name__ if frame else None
        calling_function = frame.f_code.co_name.upper() if frame else None
        msg = f"{calling_class}.{calling_function} >> {message}"
        
        # Extend logging information on debug level
        if level == logging.DEBUG:  
            calling_file = os.path.basename(frame.f_code.co_filename) if frame else None
            calling_lineno = frame.f_lineno if frame else None
            msg = f"{calling_class}.{calling_function} [{calling_file}|{calling_lineno}]>> {message}"
       
        # Call the original logging method
        self.log(level, msg, *args, **kwargs)
        return msg
    
    def getLevel(self) -> str:
        return str(logging.getLevelName(self.getEffectiveLevel())).lower()
    
    def getLevelEnum(self) -> LogLevelEnum:
        return self.LogLevelEnum(self.getLevel())

    def setLevelByEnum(self, level_enum: LogLevelEnum):
        numeric_level = logging._nameToLevel[level_enum.value]
        self.setLevel(numeric_level)

    def debug(self, message: str, frame: FrameType = None, *args, **kwargs) -> str:
        #super().debug(msg, *args, **kwargs)
        inspect_frame = frame if frame else inspect.currentframe().f_back
        msg = self._logmsg_(inspect_frame, logging.DEBUG, message)
        if self.level == logging.DEBUG:
            rc.log(f"[cyan]DEBUG:[/] {msg}")
        return f"DEBUG: {msg}"

    def info(self, message: str, frame: FrameType = None, *args, **kwargs) -> str:
        #super().info(msg, *args, **kwargs)
        inspect_frame = frame if frame else inspect.currentframe().f_back
        msg = self._logmsg_(inspect_frame, logging.INFO, message)
        if self.level == logging.DEBUG:
            rc.log(f"[green]INFO:[/] {msg}")
        return f"INFO: {msg}"

    def warning(self, message: str, frame: FrameType = None, *args, **kwargs) -> str:
        #super().warn(msg, *args, **kwargs)
        inspect_frame = frame if frame else inspect.currentframe().f_back
        msg = self._logmsg_(inspect_frame, logging.WARN, message)
        if self.level == logging.DEBUG:
            rc.log(f"[yellow]WARN:[/] {msg}")
        return f"WARN: {msg}"

    def error(self, message: str, frame: FrameType = None, *args, **kwargs) -> str:
        #super().error(msg, *args, **kwargs)
        inspect_frame = frame if frame else inspect.currentframe().f_back
        msg = self._logmsg_(inspect_frame, logging.ERROR, message)
        if self.level == logging.DEBUG:
            rc.log(f"[red]ERROR:[/] {msg}")
        return f"ERROR: {msg}"

    def critical(self, message: str, frame: FrameType = None, *args, **kwargs) -> str:
        #super().critical(msg, *args, **kwargs)
        inspect_frame = frame if frame else inspect.currentframe().f_back
        msg = self._logmsg_(inspect_frame, logging.CRITICAL, message)
        if self.level == logging.DEBUG:
            rc.log(f"[magenta]INFO:[/] {msg}")
        return f"CRITICAL: {msg}"
