# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Kokoro Text-to-Speech pipeline.

This test module validates both optimum and openvino_genai Kokoro pipelines
with a dynamically-generated tiny random model to minimize external dependencies.
"""

import json
import logging
import os
import random
import subprocess  # nosec B404
import shutil
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch

import openvino as ov
import openvino_genai as ov_genai
from optimum.intel import OVModelForTextToSpeechSeq2Seq
from utils.atomic_download import AtomicDownloadManager
from utils.constants import get_ov_cache_converted_models_dir

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
LANGUAGE = "en-us"
KOKORO_STYLE_DIM = 128
KOKORO_REF_S_DIM = 256
KOKORO_VOICE_PACK_LENGTHS = 510
KOKORO_TINY_MODEL_SEED = 1337
KOKORO_MAX_DUR = 10

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


def generate_tiny_kokoro_model(output_dir: Path) -> Path:
    """
    Generate a tiny random Kokoro model with minimal components.

    This creates a model small enough to fit in CI environments without
    requiring external model downloads. The generated model follows the
    Kokoro architecture but with reduced dimensions.

    Args:
        output_dir: Directory to save the model

    Returns:
        Path to the generated model directory
    """
    try:
        from kokoro.istftnet import Decoder
        from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder
        from transformers import AlbertConfig
    except ImportError as e:
        pytest.skip(f"Required Kokoro dependencies not available: {e}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep tiny model weights reproducible so CI does not intermittently fail
    # on unstable random checkpoints.
    random.seed(KOKORO_TINY_MODEL_SEED)
    np.random.seed(KOKORO_TINY_MODEL_SEED)
    torch.manual_seed(KOKORO_TINY_MODEL_SEED)

    # Generate vocabulary
    symbols = (
        ";:,.!?-—()'\"/ "
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "əɚɝɪʊʌæɑɔɛɜɒɹɾθðŋʃʒʤʧˈˌ"
        "àáâãäåèéêëìíîïòóôõöùúûüýÿ"
    )
    deduped = []
    seen = set()
    for ch in symbols:
        if ch not in seen:
            deduped.append(ch)
            seen.add(ch)
        if len(deduped) >= 177:
            break
    vocab = {ch: i + 1 for i, ch in enumerate(deduped)}

    # Create config
    config = {
        "model_type": "kokoro",
        "export_model_type": "kokoro",
        "hidden_dim": 512,
        "style_dim": KOKORO_STYLE_DIM,
        "n_token": 178,
        "n_layer": 1,
        "dim_in": 512,
        "n_mels": 80,
        # Keep duration head small to avoid unstable random tiny checkpoints that
        # can produce inconsistent decoder-side sequence shapes.
        "max_dur": KOKORO_MAX_DUR,
        "dropout": 0.2,
        "text_encoder_kernel_size": 3,
        "plbert": {
            "hidden_size": 128,
            "num_attention_heads": 2,
            "intermediate_size": 256,
            "max_position_embeddings": 512,
            "num_hidden_layers": 2,
            "dropout": 0.1,
        },
        "istftnet": {
            "upsample_kernel_sizes": [20, 12],
            "upsample_rates": [10, 6],
            "gen_istft_hop_size": 5,
            "gen_istft_n_fft": 20,
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "resblock_kernel_sizes": [3, 7, 11],
            "upsample_initial_channel": 512,
        },
        "vocab": vocab,
        "multispeaker": True,
        "max_conv_dim": 512,
    }

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Create and save model components
    with torch.no_grad():
        bert = CustomAlbert(AlbertConfig(vocab_size=config["n_token"], **config["plbert"]))
        bert_encoder = torch.nn.Linear(config["plbert"]["hidden_size"], config["hidden_dim"])
        predictor = ProsodyPredictor(
            style_dim=config["style_dim"],
            d_hid=config["hidden_dim"],
            nlayers=config["n_layer"],
            max_dur=config["max_dur"],
            dropout=config["dropout"],
        )
        text_encoder = TextEncoder(
            channels=config["hidden_dim"],
            kernel_size=config["text_encoder_kernel_size"],
            depth=config["n_layer"],
            n_symbols=config["n_token"],
        )
        decoder = Decoder(
            dim_in=config["hidden_dim"],
            style_dim=config["style_dim"],
            dim_out=config["n_mels"],
            **config["istftnet"],
        )

        model_path = output_dir / "tiny-kokoro-random.pth"
        torch.save(
            {
                "bert": bert.state_dict(),
                "bert_encoder": bert_encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "text_encoder": text_encoder.state_dict(),
                "decoder": decoder.state_dict(),
            },
            model_path,
        )

    # Create a deterministic Kokoro-compatible voice pack so both optimum and
    # GenAI can consume the same exported speaker data.
    voices_dir = output_dir / "voices"
    voices_dir.mkdir(exist_ok=True)
    voice_pack = torch.from_numpy(generate_speaker_embedding((KOKORO_VOICE_PACK_LENGTHS, 1, KOKORO_REF_S_DIM)))
    torch.save(voice_pack, voices_dir / "tiny_voice.pt")

    logger.info(f"Generated tiny Kokoro model at {output_dir}")
    return output_dir


def export_model_to_openvino(local_model_path: Path, output_path: Path) -> Path:
    """
    Export a Kokoro model to OpenVINO IR format using optimum-cli.

    Args:
        local_model_path: Path to local Kokoro model (with config.json)
        output_path: Directory to save OpenVINO IR artifacts

    Returns:
        Path to the converted model directory
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use optimum-cli to export
    command = [
        "optimum-cli",
        "export",
        "openvino",
        "-m",
        str(local_model_path),
        str(output_path),
        "--trust-remote-code",
    ]

    logger.info(f"Running export command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)  # nosec B603

    if result.returncode != 0:
        raise RuntimeError(
            f"Export failed with return code {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    logger.info(f"Successfully exported model to {output_path}")
    return output_path


def _ensure_tiny_voice_bin(ov_path: Path, tiny_kokoro_model_path: Path) -> Path:
    """
    Ensure the exported tiny model has a Kokoro voice pack in .bin format.

    Local-folder exports can skip voice conversion when voice files are not
    discovered by list-files logic. In that case, convert the deterministic
    source test voice pack from tiny_voice.pt to tiny_voice.bin.
    """
    voices_dir = ov_path / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)

    voice_bin = voices_dir / "tiny_voice.bin"
    if voice_bin.exists():
        return voice_bin

    source_pt = tiny_kokoro_model_path / "voices" / "tiny_voice.pt"
    if not source_pt.exists():
        raise FileNotFoundError(f"Missing source tiny voice pack for fallback conversion: {source_pt}")

    source_voice = torch.load(source_pt, map_location="cpu")
    if isinstance(source_voice, torch.Tensor):
        voice_np = source_voice.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        voice_np = np.asarray(source_voice, dtype=np.float32)

    voice_np.tofile(voice_bin)
    logger.info("Created fallback tiny voice pack at %s", voice_bin)
    return voice_bin


def generate_speaker_embedding(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Generate a deterministic speaker embedding/voice-pack for testing.

    In real scenarios, speaker embeddings come from speaker encoders or Kokoro
    voice packs. For testing, avoid random values because they can drive this
    tiny random model into unstable inference paths. Instead generate a fixed,
    bounded tensor so tests are reproducible and inputs stay well-behaved.

    Args:
        shape: Shape of the embedding tensor

    Returns:
        Deterministic float32 array with the given shape
    """
    style_dim = shape[-1]
    base = np.linspace(-0.25, 0.25, num=style_dim, dtype=np.float32)
    base = base / (np.linalg.norm(base) + 1e-8)

    if len(shape) == 3:
        # Kokoro/GenAI speaker packs are [num_lengths, 1, 256]. Reuse one stable
        # row for every possible phoneme length so length-based indexing remains valid.
        return np.broadcast_to(base.reshape(1, 1, style_dim), shape).copy()
    if len(shape) == 2:
        return np.broadcast_to(base.reshape(1, style_dim), shape).copy()
    if len(shape) == 1:
        return base.copy()

    raise ValueError(f"Unsupported speaker embedding shape: {shape}")


@pytest.fixture(scope="module")
def tiny_kokoro_model_path() -> Path:
    """
    Fixture that generates a tiny random Kokoro model for testing.

    Uses atomic download pattern for thread-safe test parallelization.
    Caches the model across test runs.
    """
    models_dir = get_ov_cache_converted_models_dir()
    model_path = models_dir / "tiny-random-kokoro"

    manager = AtomicDownloadManager(model_path)

    def _create_model_in_temp(temp_path: Path) -> None:
        generated_dir = temp_path / "model"
        generate_tiny_kokoro_model(generated_dir)

    manager.execute(_create_model_in_temp)

    # Atomic manager moves temp dir itself to final path, so files are under model/.
    nested_model_path = model_path / "model"
    if nested_model_path.exists() and (nested_model_path / "config.json").exists():
        if (model_path / "config.json").exists():
            shutil.rmtree(nested_model_path, ignore_errors=True)
        else:
            for item in nested_model_path.iterdir():
                shutil.move(str(item), str(model_path / item.name))
            shutil.rmtree(nested_model_path, ignore_errors=True)

    return model_path


@pytest.fixture(scope="module")
def tiny_kokoro_ov_path(tiny_kokoro_model_path: Path) -> Path:
    """
    Fixture that exports the tiny Kokoro model to OpenVINO format.

    Caches the exported model to avoid repeated exports in tests.
    """
    models_dir = get_ov_cache_converted_models_dir()
    ov_path = models_dir / "tiny-random-kokoro-ov"

    voice_bin = ov_path / "voices" / "tiny_voice.bin"

    # Check if already exported with the required voice pack asset.
    if not (ov_path / "openvino_model.xml").exists() or not voice_bin.exists():
        if ov_path.exists():
            shutil.rmtree(ov_path, ignore_errors=True)
        export_model_to_openvino(tiny_kokoro_model_path, ov_path)

    _ensure_tiny_voice_bin(ov_path, tiny_kokoro_model_path)

    return ov_path


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

        assert speech_optimum.ndim == 1, "Optimum speech output should be 1D"
        assert speech_genai.ndim == 1, "GenAI speech output should be 1D"
        assert speech_optimum.size > 0, "Optimum speech should not be empty"
        assert speech_genai.size > 0, "GenAI speech should not be empty"
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
