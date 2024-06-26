# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino  # add_dll_directory for openvino lib
import os
from .__version__ import __version__


if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.dirname(__file__))

from .py_generate_pipeline import (
    LLMPipeline, 
    Tokenizer, 
    GenerationConfig, 
    TokenizedInputs,
    DecodedResults, 
    EncodedResults, 
    StreamerBase, 
    StopCriteria
)
try:
    from . import py_continuous_batching
    continuous_batching = ["py_continuous_batching"]
except ImportError:
    continuous_batching = []

__all__ = [
    'LLMPipeline', 
    'Tokenizer', 
    'GenerationConfig', 
    'TokenizedInputs', 
    'DecodedResults', 
    'EncodedResults',
    'StreamerBase', 
    'StopCriteria'
] + continuous_batching
