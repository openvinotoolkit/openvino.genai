# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the Piper (VITS-family) Text-to-Speech pipeline.

Piper is a single-graph, non-autoregressive TTS model with no HuggingFace/optimum-cli
export path (unlike Kokoro/SpeechT5), so these tests use a dynamically-generated tiny
ONNX/OpenVINO fixture (see ``utils.piper_test_assets``) matching Piper's exact I/O
contract, rather than an ``optimum.intel`` model.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pytest

import openvino_genai as ov_genai
from utils.constants import get_ov_cache_converted_models_dir
from utils.piper_test_assets import prepare_tiny_piper_ov_path

logger = logging.getLogger(__name__)


def _configure_espeak_from_venv() -> None:
    """
    Point the C++ misaki espeak fallback (reused by PiperTTSImpl for raw IPA
    phonemization) at the shared library/data bundled with the ``espeakng_loader``
    Python package, for CI machines without a system-wide espeak-ng install.
    """
    try:
        import espeakng_loader  # type: ignore[import]
    except ImportError:
        logger.debug("espeakng_loader not available; skipping espeak env setup")
        return

    lib_path = espeakng_loader.get_library_path()
    data_path = espeakng_loader.get_data_path()

    if "MISAKI_ESPEAK_LIBRARY" not in os.environ:
        os.environ["MISAKI_ESPEAK_LIBRARY"] = lib_path
    if "ESPEAK_DATA_PATH" not in os.environ:
        os.environ["ESPEAK_DATA_PATH"] = data_path


_configure_espeak_from_venv()

SAMPLE_RATE = 22050


@pytest.fixture(scope="module")
def tiny_piper_ov_path() -> Path:
    """Fixture that generates a tiny random Piper OpenVINO model directory for testing."""
    models_dir = get_ov_cache_converted_models_dir()
    return prepare_tiny_piper_ov_path(models_dir)


@pytest.mark.speech_generation
class TestPiperPipeline:
    """Test suite for the Piper text-to-speech pipeline."""

    def test_dispatcher_detects_piper_backend(self, tiny_piper_ov_path: Path):
        """
        The Text2SpeechPipeline dispatcher must route a directory whose config.json
        contains a ``phoneme_id_map`` object to the Piper backend without raising.
        """
        pipe = ov_genai.Text2SpeechPipeline(str(tiny_piper_ov_path), "CPU")
        assert tuple(pipe.get_speaker_embedding_shape()) == (0,)

    def test_genai_piper_generate_produces_valid_waveform(self, tiny_piper_ov_path: Path):
        """Smoke test: a finite, non-empty 1D waveform must be produced for English text."""
        pipe = ov_genai.Text2SpeechPipeline(str(tiny_piper_ov_path), "CPU")
        result = pipe.generate("hello test", language="en-us")

        speech = result.speeches[0]
        speech_array = np.array(speech.data, dtype=np.float32).reshape(-1)

        assert result.output_sample_rate == SAMPLE_RATE
        assert speech_array.ndim == 1
        assert speech_array.size > 0
        assert np.isfinite(speech_array).all()

    def test_piper_rejects_non_empty_speaker_embedding(self, tiny_piper_ov_path: Path):
        """
        Piper voices are single-speaker; passing a non-empty speaker_embedding tensor
        must raise rather than being silently ignored.
        """
        import openvino as ov

        pipe = ov_genai.Text2SpeechPipeline(str(tiny_piper_ov_path), "CPU")
        bad_embedding = ov.Tensor(np.zeros((1, 256), dtype=np.float32))

        with pytest.raises(RuntimeError):
            pipe.generate("hello test", bad_embedding, language="en-us")

    def test_piper_output_length_scales_with_text_length(self, tiny_piper_ov_path: Path):
        """
        Longer input text should map to a longer phoneme-id sequence, and therefore to a
        longer waveform for this fixture's length-proportional structural graph.
        """
        pipe = ov_genai.Text2SpeechPipeline(str(tiny_piper_ov_path), "CPU")

        short_result = pipe.generate("ab", language="en-us")
        long_result = pipe.generate("ab ab ab ab ab", language="en-us")

        short_len = np.array(short_result.speeches[0].data).size
        long_len = np.array(long_result.speeches[0].data).size

        assert long_len > short_len

    def test_piper_scales_are_configurable(self, tiny_piper_ov_path: Path):
        """noise_scale/length_scale/noise_w must be accepted as generation properties."""
        pipe = ov_genai.Text2SpeechPipeline(str(tiny_piper_ov_path), "CPU")
        result = pipe.generate(
            "hello test",
            language="en-us",
            noise_scale=0.5,
            length_scale=1.2,
            noise_w=0.6,
        )
        speech_array = np.array(result.speeches[0].data, dtype=np.float32).reshape(-1)
        assert speech_array.size > 0
        assert np.isfinite(speech_array).all()
