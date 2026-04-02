# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import sys


def install_av_stub_module_for_windows():
    """Install a stub av module on Windows to avoid PyAV/ffmpeg DLL loading issues.
    Transformers pipeline may import VideoClassificationPipeline, which depends on PyAV.
    Workaround for ticket CVS-179943.
    This workaround is also needed for videochat-flash-qwen model tests due to the same
    PyAV dependency, see CVS-183222.
    """
    if sys.platform != "win32":
        return

    from types import ModuleType

    sys.modules["av"] = ModuleType("av")
    sys.modules["av"].__version__ = "0.0.0"
