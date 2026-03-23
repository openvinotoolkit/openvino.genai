# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import sys


def install_av_stub_module_for_windows():
    # win32 fails on ffmpeg DLLs load
    # import transformers.pipeline imports VideoClassificationPipeline which requires PyAV (ffmpeg bindings)
    # wa is to create mock 'av' module to prevent DLLs loading errors
    # ticket: 179943
    if sys.platform != "win32":
        return

    from types import ModuleType

    sys.modules["av"] = ModuleType("av")
    sys.modules["av"].__version__ = "0.0.0"


if __name__ == "__main__":
    install_av_stub_module_for_windows()
