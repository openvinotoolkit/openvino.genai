# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openvino genai module namespace, exposing pipelines and configs to create these pipelines."""

import os

import openvino  # add_dll_directory for openvino lib

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.dirname(__file__))

from .py_openvino_genai import (
    DecodedResults,
    EncodedResults,
    RawPerfMetrics,
    PerfMetrics,
    StreamerBase,
    OmniSpeechStreamerBase,
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
    VLMPipelineBase,
    VLMDecodedResults,
    VideoMetadata,
)

# Omni pipeline (Qwen3-Omni text + speech)

from .py_openvino_genai import (
    OmniDecodedResults,
    OmniPipeline,
    OmniTalkerSpeechConfig,
    Talker,
    TalkerBase,
    TalkerPerfMetrics,
    TalkerResults,
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

# ASR pipeline
from .py_openvino_genai import (
    ASRDecodedResultChunk,
    ASRDecodedResults,
    ASRGenerationConfig,
    ASRPerfMetrics,
    ASRPipeline,
    ASRRawPerfMetrics,
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
    Qwen3TextEncoder,
    T5EncoderModel,
    UNet2DConditionModel,
    Flux2Transformer2DModel,
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
from .py_openvino_genai import EmbedResult, EmbeddingPipeline, TextEmbeddingPipeline, TextRerankPipeline

# Speech generation
from .py_openvino_genai import (
    SpeechGenerationConfig,
    SpeechGenerationPerfMetrics,
    Text2SpeechDecodedResults,
    Text2SpeechPipeline,
)
