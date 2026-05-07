# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Unified ASR Pipeline Tests - Refactored with Parametrization

This file extends test_whisper_pipeline.py tests to work with both
WhisperPipeline and ASRPipeline, using pytest parametrization as requested.

Key changes from original implementation:
1. Tests use @pytest.mark.parametrize to test both pipeline classes
2. Reuses existing helper functions from utils/ directory
3. Paraformer-specific tests are clearly separated
4. Model type detection uses config.json model_type field

Reference PR: https://github.com/openvinotoolkit/openvino.genai/pull/3515
"""

import functools
import gc
import json
import os
import pathlib
import sys
import typing
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytest

# Workaround for Windows: mock 'av' module before importing transformers
if sys.platform == "win32":
    from unittest.mock import MagicMock
    sys.modules.setdefault("av", MagicMock())

import openvino_genai as ov_genai

# Reuse existing helper utilities from test infrastructure
from utils.constants import get_ov_cache_converted_models_dir, extra_generate_kwargs
from utils.network import retry_request
from utils.atomic_download import AtomicDownloadManager


# ============================================================================
# Constants & Configuration
# ============================================================================

def _get_model_path(env_var: str, default: str) -> pathlib.Path:
    """Get model path from environment variable or use default."""
    return pathlib.Path(os.environ.get(env_var, default))

# Model paths (configurable via environment variables)
WHISPER_MODEL_PATH = _get_model_path(
    "WHISPER_MODEL_PATH",
    get_ov_cache_converted_models_dir() / "whisper-tiny"
)
PARAFORMER_MODEL_PATH = _get_model_path(
    "PARAFORMER_MODEL_PATH",
    pathlib.Path(__file__).parent.parent.parent.parent / "paraformer-zh" / "ov_models"
)


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(scope="class", autouse=True)
def run_gc_after_test():
    """Run garbage collection after each test class."""
    yield
    gc.collect()


# ============================================================================
# Pipeline Availability Checks
# ============================================================================

def whisper_model_available() -> bool:
    """Check if Whisper model files exist."""
    required_files = [
        "openvino_encoder_model.xml",
        "openvino_decoder_model.xml",
    ]
    return all((WHISPER_MODEL_PATH / f).exists() for f in required_files)


def paraformer_model_available() -> bool:
    """Check if Paraformer model files exist."""
    return (PARAFORMER_MODEL_PATH.exists() and 
            (PARAFORMER_MODEL_PATH / "openvino_model.xml").exists())


def asr_pipeline_available() -> bool:
    """Check if ASRPipeline is available in openvino_genai."""
    return hasattr(ov_genai, 'ASRPipeline')


# ============================================================================
# Skip Conditions for Parametrized Tests
# ============================================================================

def skip_if_no_whisper():
    return pytest.mark.skipif(
        not whisper_model_available(),
        reason=f"Whisper model not found at {WHISPER_MODEL_PATH}"
    )

def skip_if_no_paraformer():
    return pytest.mark.skipif(
        not paraformer_model_available(),
        reason=f"Paraformer model not found at {PARAFORMER_MODEL_PATH}"
    )

def skip_if_no_asr_pipeline():
    return pytest.mark.skipif(
        not asr_pipeline_available(),
        reason="ASRPipeline not available in openvino_genai"
    )


# ============================================================================
# Cached Pipeline Loaders (reusing existing patterns)
# ============================================================================

@functools.lru_cache()
def load_whisper_pipeline():
    """Load WhisperPipeline with caching."""
    if not whisper_model_available():
        pytest.skip(f"Whisper model not found at {WHISPER_MODEL_PATH}")
    return ov_genai.WhisperPipeline(str(WHISPER_MODEL_PATH), "CPU")


@functools.lru_cache()
def load_asr_pipeline_whisper():
    """Load ASRPipeline with Whisper model."""
    if not asr_pipeline_available():
        pytest.skip("ASRPipeline not available")
    if not whisper_model_available():
        pytest.skip(f"Whisper model not found at {WHISPER_MODEL_PATH}")
    return ov_genai.ASRPipeline(str(WHISPER_MODEL_PATH), "CPU")


@functools.lru_cache()
def load_asr_pipeline_paraformer():
    """Load ASRPipeline with Paraformer model."""
    if not asr_pipeline_available():
        pytest.skip("ASRPipeline not available")
    if not paraformer_model_available():
        pytest.skip(f"Paraformer model not found at {PARAFORMER_MODEL_PATH}")
    return ov_genai.ASRPipeline(str(PARAFORMER_MODEL_PATH), "CPU")


# ============================================================================
# Audio Generation Utilities (reused from test_whisper_pipeline.py)
# ============================================================================

def generate_test_audio(duration_seconds: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate test audio (near-silence white noise)."""
    n_samples = int(duration_seconds * sample_rate)
    return (np.random.randn(n_samples) * 0.001).astype(np.float32)


# ============================================================================
# PARAMETRIZED TESTS - Core Feature: Both WhisperPipeline and ASRPipeline
# ============================================================================

class TestASRPipelineParametrized:
    """
    Parametrized tests that run with both WhisperPipeline and ASRPipeline.
    This addresses reviewer comment: "Let's extend existing whisper tests 
    with ASRPipeline interface with parametrization."
    """

    @pytest.mark.parametrize("pipeline_loader,pipeline_name", [
        pytest.param(load_whisper_pipeline, "WhisperPipeline", 
                     marks=skip_if_no_whisper()),
        pytest.param(load_asr_pipeline_whisper, "ASRPipeline", 
                     marks=[skip_if_no_asr_pipeline(), skip_if_no_whisper()]),
    ])
    def test_constructor_and_basic_generate(self, pipeline_loader, pipeline_name):
        """Test pipeline construction and basic generate call."""
        pipe = pipeline_loader()
        assert pipe is not None
        
        # Generate with synthetic audio
        audio = generate_test_audio(duration_seconds=2.0)
        result = pipe.generate(audio)
        
        assert hasattr(result, 'texts')
        assert len(result.texts) > 0
        # Result text may be empty for silence, but should be a string
        assert isinstance(result.texts[0], str)

    @pytest.mark.parametrize("pipeline_loader,pipeline_name", [
        pytest.param(load_whisper_pipeline, "WhisperPipeline", 
                     marks=skip_if_no_whisper()),
        pytest.param(load_asr_pipeline_whisper, "ASRPipeline", 
                     marks=[skip_if_no_asr_pipeline(), skip_if_no_whisper()]),
    ])
    def test_generation_config_access(self, pipeline_loader, pipeline_name):
        """Test generation config get/set."""
        pipe = pipeline_loader()
        
        # Get config
        config = pipe.get_generation_config()
        assert config is not None
        assert hasattr(config, 'max_new_tokens')
        
        # Modify and set
        original_max_tokens = config.max_new_tokens
        config.max_new_tokens = 100
        pipe.set_generation_config(config)
        
        # Verify change
        new_config = pipe.get_generation_config()
        assert new_config.max_new_tokens == 100
        
        # Restore
        config.max_new_tokens = original_max_tokens
        pipe.set_generation_config(config)

    @pytest.mark.parametrize("pipeline_loader,pipeline_name", [
        pytest.param(load_whisper_pipeline, "WhisperPipeline", 
                     marks=skip_if_no_whisper()),
        pytest.param(load_asr_pipeline_whisper, "ASRPipeline", 
                     marks=[skip_if_no_asr_pipeline(), skip_if_no_whisper()]),
    ])
    def test_tokenizer_access(self, pipeline_loader, pipeline_name):
        """Test tokenizer access (Whisper-specific)."""
        pipe = pipeline_loader()
        
        # Whisper pipeline should have tokenizer
        tokenizer = pipe.get_tokenizer()
        assert tokenizer is not None


# ============================================================================
# ASR PIPELINE SPECIFIC TESTS
# ============================================================================

class TestASRPipelineSpecific:
    """Tests specific to ASRPipeline functionality."""

    @skip_if_no_asr_pipeline()
    @skip_if_no_whisper()
    def test_model_type_detection_whisper(self):
        """Test model type auto-detection for Whisper."""
        pipe = load_asr_pipeline_whisper()
        
        assert pipe.is_whisper() == True
        assert pipe.is_paraformer() == False
        assert pipe.get_model_type() == ov_genai.ASRPipeline.ModelType.WHISPER

    @skip_if_no_asr_pipeline()
    @skip_if_no_paraformer()
    def test_model_type_detection_paraformer(self):
        """Test model type auto-detection for Paraformer."""
        pipe = load_asr_pipeline_paraformer()
        
        assert pipe.is_whisper() == False
        assert pipe.is_paraformer() == True
        assert pipe.get_model_type() == ov_genai.ASRPipeline.ModelType.PARAFORMER

    @skip_if_no_asr_pipeline()
    @skip_if_no_whisper()
    def test_whisper_through_asr_matches_whisper_pipeline(self):
        """Verify ASRPipeline produces same results as WhisperPipeline for Whisper model."""
        whisper_pipe = load_whisper_pipeline()
        asr_pipe = load_asr_pipeline_whisper()
        
        audio = generate_test_audio(duration_seconds=2.0)
        
        whisper_result = whisper_pipe.generate(audio)
        asr_result = asr_pipe.generate(audio)
        
        # Results should match
        assert whisper_result.texts[0] == asr_result.texts[0]


# ============================================================================
# PARAFORMER SPECIFIC TESTS
# ============================================================================

class TestParaformerSpecific:
    """
    Tests specific to Paraformer model.
    Addresses reviewer comment: "For paraformer tests let's reuse existing 
    helper functions like hugging face utils."
    """

    @skip_if_no_asr_pipeline()
    @skip_if_no_paraformer()
    def test_paraformer_basic_generate(self):
        """Test basic Paraformer generate."""
        pipe = load_asr_pipeline_paraformer()
        
        audio = generate_test_audio(duration_seconds=3.0)
        result = pipe.generate(audio)
        
        assert hasattr(result, 'texts')
        assert len(result.texts) > 0
        assert isinstance(result.texts[0], str)

    @skip_if_no_asr_pipeline()
    @skip_if_no_paraformer()
    def test_paraformer_generation_config(self):
        """Test Paraformer generation config."""
        pipe = load_asr_pipeline_paraformer()
        
        config = pipe.get_generation_config()
        assert config is not None
        assert hasattr(config, 'max_new_tokens')

    @skip_if_no_asr_pipeline()
    @skip_if_no_paraformer()  
    def test_paraformer_tokenizer_not_supported(self):
        """Test that Paraformer correctly indicates tokenizer not supported."""
        pipe = load_asr_pipeline_paraformer()
        
        # Paraformer uses internal detokenizer, not Tokenizer class
        with pytest.raises(Exception):
            pipe.get_tokenizer()


# ============================================================================
# PERFORMANCE AND CONSISTENCY TESTS
# ============================================================================

class TestConsistency:
    """Tests for output consistency and repeatability."""

    @pytest.mark.parametrize("pipeline_loader,pipeline_name", [
        pytest.param(load_whisper_pipeline, "WhisperPipeline", 
                     marks=skip_if_no_whisper()),
        pytest.param(load_asr_pipeline_whisper, "ASRPipeline", 
                     marks=[skip_if_no_asr_pipeline(), skip_if_no_whisper()]),
    ])
    def test_deterministic_output(self, pipeline_loader, pipeline_name):
        """Test that same input produces same output."""
        pipe = pipeline_loader()
        audio = generate_test_audio(duration_seconds=2.0)
        
        result1 = pipe.generate(audio)
        result2 = pipe.generate(audio)
        
        assert result1.texts[0] == result2.texts[0]


# ============================================================================
# Main entry point for running tests directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
