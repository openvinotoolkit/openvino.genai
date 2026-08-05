# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Kokoro Text-to-Speech pipeline.

This test module validates both optimum and openvino_genai Kokoro pipelines
with a dynamically-generated tiny random model to minimize external dependencies.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pytest

import openvino as ov
import openvino_genai as ov_genai
from optimum.intel import OVModelForTextToSpeechSeq2Seq
from utils.constants import get_ov_cache_converted_models_dir
from utils.kokoro_test_assets import prepare_tiny_g2p_model_path
from utils.kokoro_test_assets import prepare_tiny_g2p_ov_path
from utils.kokoro_test_assets import prepare_tiny_kokoro_model_path
from utils.kokoro_test_assets import prepare_tiny_kokoro_ov_path

logger = logging.getLogger(__name__)


def _configure_espeak_from_venv() -> None:
    """
    Point the C++ misaki espeak fallback at the DLL and data directory that
    come bundled with the ``espeakng_loader`` Python package (installed as a
    transitive dependency of ``misaki``).  This is necessary on CI machines
    where espeak-ng is not installed in a system location.

    Sets two environment variables **only if they are not already set**:
    * ``MISAKI_ESPEAK_LIBRARY`` – absolute path to the espeak-ng shared library
      (read by ``kokoro_tts_model.cpp`` via ``std::getenv``).
    * ``ESPEAK_DATA_PATH`` – directory containing ``espeak-ng-data/``
      (read by the espeak-ng runtime itself during ``espeak_Initialize``).
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
        logger.debug("Set MISAKI_ESPEAK_LIBRARY=%s", lib_path)

    if "ESPEAK_DATA_PATH" not in os.environ:
        os.environ["ESPEAK_DATA_PATH"] = data_path
        logger.debug("Set ESPEAK_DATA_PATH=%s", data_path)


# Configure espeak at import time so the env vars are in place before the
# first openvino_genai.Text2SpeechPipeline is constructed.
_configure_espeak_from_venv()


SAMPLE_RATE = 24000

MULTILINGUAL_PROMPT_CASES = [
    ("Hello this is a short speech generation test.", "en-us"),
    ("Today we analyse colour and flavour.", "en-gb"),
    ("Hola esto es una prueba de voz sintetica.", "es"),
    ("Bonjour ceci est un test de synthese vocale.", "fr-fr"),
    ("नमस्ते यह वाक संश्लेषण परीक्षण है।", "hi"),
    ("Ciao questo e un test di sintesi vocale.", "it"),
    ("Ola isto e um teste de sintese de voz.", "pt-br"),
]
MULTILINGUAL_CASE_IDS = [f"multilingual-{language}" for _, language in MULTILINGUAL_PROMPT_CASES]


def _to_optimum_lang_code(language: str) -> str:
    """Map GenAI language variants to Kokoro KPipeline lang_code values."""
    mapping = {
        "en-us": "a",
        "en-gb": "b",
    }
    return mapping.get(language, language)


@pytest.fixture(scope="module")
def tiny_kokoro_model_path() -> Path:
    """
    Fixture that generates a tiny random Kokoro model for testing.

    Uses atomic download pattern for thread-safe test parallelization.
    Caches the model across test runs.
    """
    models_dir = get_ov_cache_converted_models_dir()
    return prepare_tiny_kokoro_model_path(models_dir)


@pytest.fixture(scope="module")
def tiny_kokoro_ov_path(tiny_kokoro_model_path: Path) -> Path:
    """
    Fixture that exports the tiny Kokoro model to OpenVINO format.

    Caches the exported model to avoid repeated exports in tests.
    """
    models_dir = get_ov_cache_converted_models_dir()
    return prepare_tiny_kokoro_ov_path(models_dir, tiny_kokoro_model_path)


@pytest.fixture
def speaker_embedding_tensor(tiny_kokoro_ov_path: Path):
    """
    Fixture that provides a speaker embedding tensor suitable for inference.

    For the tiny model, load the exported deterministic voice pack so the same
    speaker data is used by both optimum and GenAI.
    """
    pipe = ov_genai.Text2SpeechPipeline(str(tiny_kokoro_ov_path), "CPU")
    shape = tuple(pipe.get_speaker_embedding_shape())
    voice_bin = tiny_kokoro_ov_path / "voices" / "tiny_voice.bin"
    if not voice_bin.exists():
        raise FileNotFoundError(f"Expected exported Kokoro voice pack at {voice_bin}")

    embedding_array = np.fromfile(voice_bin, dtype=np.float32)
    expected_size = int(np.prod(shape))
    if embedding_array.size != expected_size:
        raise ValueError(
            f"Exported voice pack size mismatch: got {embedding_array.size}, expected {expected_size} for shape {shape}"
        )

    return ov.Tensor(embedding_array.reshape(shape))


@pytest.fixture(scope="module")
def tiny_g2p_model_path() -> Path:
    """
    Generate a tiny random BART G2P model and cache it across the test session.

    Follows the same AtomicDownloadManager pattern used for the Kokoro model
    so parallel test workers do not race on the output directory.
    """
    models_dir = get_ov_cache_converted_models_dir()
    return prepare_tiny_g2p_model_path(models_dir)


@pytest.fixture(scope="module")
def tiny_g2p_ov_path(tiny_g2p_model_path: Path) -> Path:
    """
    Export the tiny G2P BART model to OpenVINO IR and cache the result.

    Produces ``openvino_encoder_model.xml`` and ``openvino_decoder_model.xml``
    in a sibling directory to ``tiny_g2p_model_path``.
    """
    models_dir = get_ov_cache_converted_models_dir()
    return prepare_tiny_g2p_ov_path(models_dir, tiny_g2p_model_path)


@pytest.mark.speech_generation
class TestKokoroPipeline:
    """Test suite for Kokoro text-to-speech pipeline."""

    @pytest.mark.parametrize("prompt,language", MULTILINGUAL_PROMPT_CASES, ids=MULTILINGUAL_CASE_IDS)
    def test_genai_kokoro_prompt_language_matrix(
        self,
        tiny_kokoro_ov_path: Path,
        speaker_embedding_tensor: ov.Tensor,
        prompt: str,
        language: str,
    ):
        """
        Validate Kokoro GenAI inference across several prompt/language pairs.

        This is a smoke-coverage test: each case must produce a finite, non-empty
        1D waveform. Exact sample-by-sample parity is intentionally not required
        here; that is covered by the dedicated consistency test.
        """
        pipe = ov_genai.Text2SpeechPipeline(str(tiny_kokoro_ov_path), "CPU")
        result = pipe.generate(prompt, speaker_embedding_tensor, language=language)

        speech = result.speeches[0]
        speech_array = np.array(speech.data, dtype=np.float32).reshape(-1)

        assert result.output_sample_rate == SAMPLE_RATE, (
            f"Expected sample rate {SAMPLE_RATE}, got {result.output_sample_rate}"
        )
        assert speech_array.ndim == 1, "Speech output should be 1D array"
        assert speech_array.size > 0, "Speech should not be empty"
        assert np.isfinite(speech_array).all(), "Speech should contain no NaN or Inf"
        assert np.std(speech_array) > 0.005, "Speech output should have non-trivial variance"

        logger.info(
            "Case language=%s prompt=%r -> shape=%s range=[%.4f, %.4f]",
            language,
            prompt,
            speech_array.shape,
            float(speech_array.min()),
            float(speech_array.max()),
        )

    @pytest.mark.parametrize("prompt,language", MULTILINGUAL_PROMPT_CASES, ids=MULTILINGUAL_CASE_IDS)
    def test_kokoro_output_consistency_prompt_language_matrix(
        self,
        tiny_kokoro_ov_path: Path,
        speaker_embedding_tensor: ov.Tensor,
        prompt: str,
        language: str,
    ):
        """
        Exact-match parity test across multiple prompt/language pairs.

        This extends the single-case consistency test to verify that optimum and
        GenAI produce identical waveforms for the same prompt, language, and
        speaker embedding.
        """
        # Optimum path with explicit language mapping for Kokoro KPipeline.
        model = OVModelForTextToSpeechSeq2Seq.from_pretrained(str(tiny_kokoro_ov_path), trust_remote_code=True)
        inputs = model.preprocess_input(
            text=prompt,
            speaker_embedding=speaker_embedding_tensor,
            lang_code=_to_optimum_lang_code(language),
        )
        speech_optimum = model.generate(**inputs).numpy()

        # GenAI path with language variant string.
        pipe = ov_genai.Text2SpeechPipeline(str(tiny_kokoro_ov_path), "CPU")
        result = pipe.generate(prompt, speaker_embedding_tensor, language=language)
        speech_genai = np.array(result.speeches[0].data, dtype=np.float32).reshape(-1)

        assert np.isfinite(speech_optimum).all(), "Optimum speech should contain no NaN/Inf"
        assert np.isfinite(speech_genai).all(), "GenAI speech should contain no NaN/Inf"
        assert speech_optimum.shape == speech_genai.shape, (
            f"Output shapes differ for language={language!r}, prompt={prompt!r}: "
            f"optimum={speech_optimum.shape}, genai={speech_genai.shape}"
        )

        if not np.array_equal(speech_optimum, speech_genai):
            diff_indices = np.flatnonzero(speech_optimum != speech_genai)
            first_diff = int(diff_indices[0]) if diff_indices.size else -1
            max_diff = float(np.max(np.abs(speech_optimum - speech_genai)))
            mean_diff = float(np.mean(np.abs(speech_optimum - speech_genai)))
            pytest.fail(
                "Outputs are not an exact match for "
                f"language={language!r}, prompt={prompt!r}: "
                f"first_diff_index={first_diff}, "
                f"optimum={speech_optimum[first_diff] if first_diff >= 0 else 'n/a'}, "
                f"genai={speech_genai[first_diff] if first_diff >= 0 else 'n/a'}, "
                f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
            )


@pytest.mark.speech_generation
class TestKokoroFallback:
    """
    Tests for the OpenVINO G2P fallback mechanism (``phonemize_fallback_model_dir``).

    The fallback lets callers supply a small BART-based grapheme-to-phoneme model
    that is invoked for out-of-vocabulary tokens the primary misaki G2P engine
    cannot handle.  These tests exercise that code path using a tiny random
    model whose predicted phonemes are nonsensical but whose inference must
    complete without errors.

    Because the fallback is only wired for English variants, all prompts here
    use ``language="en-us"``.
    """

    def test_fallback_produces_valid_speech(
        self,
        tiny_kokoro_ov_path: Path,
        tiny_g2p_ov_path: Path,
        speaker_embedding_tensor: ov.Tensor,
    ):
        """
        Smoke test: configure the OV G2P fallback and verify the pipeline
        produces a finite, non-empty waveform.

        The tiny random G2P model returns garbage phonemes for any token it
        processes, but the end-to-end pipeline must complete without errors
        and the resulting audio must pass basic sanity checks.

        Note: optimum-intel currently does not support this fallback mechanism,
        so this is why we don't perform a direct optimum-vs-GenAI check here.
        """
        pipe = ov_genai.Text2SpeechPipeline(str(tiny_kokoro_ov_path), "CPU")
        result = pipe.generate(
            # Fictional words are absent from the misaki English lexicon
            # and will therefore trigger the OV fallback.
            "Vellorin traded copperchimes for rainmint at Candlehaven.",
            speaker_embedding_tensor,
            language="en-us",
            phonemize_fallback_model_dir=str(tiny_g2p_ov_path),
        )

        speech_array = np.array(result.speeches[0].data, dtype=np.float32).reshape(-1)

        assert speech_array.ndim == 1
        assert speech_array.size > 0, "Speech output should not be empty"
        assert np.isfinite(speech_array).all(), "Speech output should contain no NaN or Inf"

    def test_fallback_does_not_change_in_vocab_output(
        self,
        tiny_kokoro_ov_path: Path,
        tiny_g2p_ov_path: Path,
        speaker_embedding_tensor: ov.Tensor,
    ):
        """
        Verify that configuring the OV fallback does not alter the output for
        text whose tokens are all handled by misaki's built-in English lexicon.

        The fallback hook is only invoked for OOV tokens.  For fully in-vocabulary
        text the phoneme sequence is identical regardless of fallback configuration,
        so the synthesised waveforms must be bit-exact.
        """
        prompt = "The quick brown fox."

        pipe_no_fb = ov_genai.Text2SpeechPipeline(str(tiny_kokoro_ov_path), "CPU")
        result_no_fb = pipe_no_fb.generate(prompt, speaker_embedding_tensor, language="en-us")
        speech_no_fb = np.array(result_no_fb.speeches[0].data, dtype=np.float32).reshape(-1)

        pipe_fb = ov_genai.Text2SpeechPipeline(str(tiny_kokoro_ov_path), "CPU")
        result_fb = pipe_fb.generate(
            prompt,
            speaker_embedding_tensor,
            language="en-us",
            phonemize_fallback_model_dir=str(tiny_g2p_ov_path),
        )
        speech_fb = np.array(result_fb.speeches[0].data, dtype=np.float32).reshape(-1)

        assert speech_no_fb.size > 0
        assert speech_fb.size > 0
        assert np.isfinite(speech_no_fb).all()
        assert np.isfinite(speech_fb).all()
        assert np.array_equal(speech_no_fb, speech_fb), (
            "Speech output should be bit-exact for in-vocabulary text "
            "regardless of whether the OV G2P fallback is configured"
        )
