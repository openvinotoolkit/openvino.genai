# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions and classes for OpenVINO GenAI ModulePipeline tools.
"""

# ============================================================================
# Logging
# ============================================================================

class Logger:
    """Simple logger with configurable verbosity levels."""
    
    QUIET = 0
    ERROR = 1
    INFO = 2
    DEBUG = 3

    def __init__(self, level=INFO):
        self.level = level

    def error(self, msg):
        if self.level >= self.ERROR:
            print(f"[ERROR] {msg}")

    def info(self, msg):
        if self.level >= self.INFO:
            print(f"[INFO] {msg}")

    def debug(self, msg):
        if self.level >= self.DEBUG:
            print(f"[DEBUG] {msg}")

    def success(self, msg):
        if self.level >= self.INFO:
            print(f"[SUCCESS] {msg}")

    def warning(self, msg):
        if self.level >= self.INFO:
            print(f"[WARNING] {msg}")
