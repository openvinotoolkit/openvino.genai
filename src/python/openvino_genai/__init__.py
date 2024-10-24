# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openvino genai module namespace, exposing pipelines and configs to create these pipelines."""

import openvino  # add_dll_directory for openvino lib
import os
from .__version__ import __version__


if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.dirname(__file__))

from .py_openvino_genai import (
    ContinuousBatchingPipeline,
    DecodedResults,
    EncodedResults,
    GenerationConfig,
    GenerationResult,
    Adapter,
    AdapterConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    UNet2DConditionModel,
    AutoencoderKL,
    LLMPipeline, 
    VLMPipeline,
    Text2ImagePipeline,
    PerfMetrics,
    RawPerfMetrics,
    SchedulerConfig,
    StopCriteria,
    StreamerBase,
    TokenizedInputs,
    Tokenizer,
    WhisperGenerationConfig,
    WhisperPipeline,
    CacheEvictionConfig,
    AggregationMode,
    Generator,
    CppStdGenerator,
    draft_model,
    CandidatesMathingType
)
