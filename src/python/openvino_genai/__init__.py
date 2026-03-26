# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openvino genai module namespace, exposing pipelines and configs to create these pipelines."""

import os

import openvino  # noqa: F401  # add_dll_directory for openvino lib

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.dirname(__file__))

from .py_openvino_genai import (
    DecodedResults,
    EncodedResults,
    RawPerfMetrics,
    PerfMetrics,
    StreamerBase,
    get_version,
    StreamingStatus,
    TextStreamer,
    TextParserStreamer,
)

from .py_openvino_genai import (
    Parser,
    VLLMParserWrapper,
    ReasoningParser,
    DeepSeekR1ReasoningParser,
    Phi4ReasoningParser,
    Llama3JsonToolParser,
    Llama3PythonicToolParser,
    IncrementalParser,
    ReasoningIncrementalParser,
    DeepSeekR1ReasoningIncrementalParser,
    Phi4ReasoningIncrementalParser,
)

__version__ = get_version()

# VLM pipeline

from .py_openvino_genai import (
    VLMPipeline,
)

# LLM pipeline
from .py_openvino_genai import (
    LLMPipeline,
    draft_model,
)

# LoRA
from .py_openvino_genai import Adapter, AdapterConfig

# Generation config
from .py_openvino_genai import (
    GenerationConfig,
    StructuralTagItem,
    StructuralTagsConfig,
    StructuredOutputConfig,
    StopCriteria,
)

# Chat history
from .py_openvino_genai import ChatHistory

# Tokenizers
from .py_openvino_genai import TokenizedInputs, Tokenizer

# Whisper
from .py_openvino_genai import (
    WhisperGenerationConfig,
    WhisperPipeline,
    WhisperRawPerfMetrics,
    WhisperPerfMetrics,
    WhisperWordTiming,
)

# Image generation
from .py_openvino_genai import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    T5EncoderModel,
    UNet2DConditionModel,
    FluxTransformer2DModel,
    SD3Transformer2DModel,
    AutoencoderKL,
    Text2ImagePipeline,
    Image2ImagePipeline,
    InpaintingPipeline,
    Scheduler,
    ImageGenerationConfig,
    Generator,
    CppStdGenerator,
    TorchGenerator,
    ImageGenerationPerfMetrics,
    RawImageGenerationPerfMetrics,
    TaylorSeerCacheConfig,
)

# Video generation
from .py_openvino_genai import (
    LTXVideoTransformer3DModel,
    AutoencoderKLLTXVideo,
    Text2VideoPipeline,
    VideoGenerationConfig,
    VideoGenerationResult,
    VideoGenerationPerfMetrics,
)

# Continuous batching
from .py_openvino_genai import (
    ContinuousBatchingPipeline,
    GenerationFinishReason,
    GenerationResult,
    GenerationStatus,
    SchedulerConfig,
    CacheEvictionConfig,
    AggregationMode,
    SparseAttentionMode,
    SparseAttentionConfig,
    KVCrushAnchorPointMode,
    KVCrushConfig,
)

# RAG
from .py_openvino_genai import TextEmbeddingPipeline, TextRerankPipeline

# Speech generation
from .py_openvino_genai import (
    SpeechGenerationConfig,
    SpeechGenerationPerfMetrics,
    Text2SpeechDecodedResults,
    Text2SpeechPipeline,
)

__all__ = [
    "Adapter",
    "AdapterConfig",
    "AggregationMode",
    "AutoencoderKL",
    "AutoencoderKLLTXVideo",
    "CacheEvictionConfig",
    "ChatHistory",
    "CLIPTextModel",
    "CLIPTextModelWithProjection",
    "ContinuousBatchingPipeline",
    "CppStdGenerator",
    "DecodedResults",
    "DeepSeekR1ReasoningIncrementalParser",
    "DeepSeekR1ReasoningParser",
    "draft_model",
    "EncodedResults",
    "FluxTransformer2DModel",
    "GenerationConfig",
    "GenerationFinishReason",
    "GenerationResult",
    "GenerationStatus",
    "Generator",
    "Image2ImagePipeline",
    "ImageGenerationConfig",
    "ImageGenerationPerfMetrics",
    "IncrementalParser",
    "InpaintingPipeline",
    "KVCrushAnchorPointMode",
    "KVCrushConfig",
    "LLMPipeline",
    "Llama3JsonToolParser",
    "Llama3PythonicToolParser",
    "LTXVideoTransformer3DModel",
    "Parser",
    "PerfMetrics",
    "Phi4ReasoningIncrementalParser",
    "Phi4ReasoningParser",
    "RawImageGenerationPerfMetrics",
    "RawPerfMetrics",
    "ReasoningIncrementalParser",
    "ReasoningParser",
    "SD3Transformer2DModel",
    "Scheduler",
    "SchedulerConfig",
    "SparseAttentionConfig",
    "SparseAttentionMode",
    "SpeechGenerationConfig",
    "SpeechGenerationPerfMetrics",
    "StopCriteria",
    "StreamerBase",
    "StreamingStatus",
    "StructuralTagItem",
    "StructuralTagsConfig",
    "StructuredOutputConfig",
    "TaylorSeerCacheConfig",
    "T5EncoderModel",
    "Text2ImagePipeline",
    "Text2SpeechDecodedResults",
    "Text2SpeechPipeline",
    "Text2VideoPipeline",
    "TextEmbeddingPipeline",
    "TextParserStreamer",
    "TextRerankPipeline",
    "TextStreamer",
    "TokenizedInputs",
    "Tokenizer",
    "TorchGenerator",
    "UNet2DConditionModel",
    "VideoGenerationConfig",
    "VideoGenerationPerfMetrics",
    "VideoGenerationResult",
    "VLMPipeline",
    "VLLMParserWrapper",
    "WhisperGenerationConfig",
    "WhisperPerfMetrics",
    "WhisperPipeline",
    "WhisperRawPerfMetrics",
    "WhisperWordTiming",
    "get_version",
]
