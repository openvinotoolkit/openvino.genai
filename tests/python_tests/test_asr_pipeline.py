# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Unified ASR Pipeline Tests for Whisper and Paraformer models.

This test file provides feature parity with test_whisper_pipeline.py for both
Whisper and Paraformer models:

  Whisper tests (mirrors every test in test_whisper_pipeline.py):
    - Smoke / constructor tests (with and without HuggingFace reference comparison)
    - Generation config tests
    - Timestamp tests (short-form + max_new_tokens)
    - Language / task / autodetect tests
    - Long-form and short-form audio tests
    - Beam search and random sampling
    - Streamer tests (6 streamer types)
    - Detailed performance metrics

  Paraformer tests (analogous subset where applicable):
    - Smoke / constructor/positional/variation tests
    - Short-form and long-form audio
    - max_new_tokens passthrough
    - Consistency and detailed performance metrics

Reference: https://github.com/openvinotoolkit/openvino.genai/pull/2804
"""

import functools
import gc
import importlib.metadata as metadata
import json
import os
import pathlib
import sys
import time
import typing
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from packaging.version import parse
from unittest.mock import MagicMock

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
import pytest

# Workaround for Windows: mock 'av' module before importing transformers.pipeline
# This mirrors test_whisper_pipeline.py approach to avoid import errors on win32.
if sys.platform == "win32":
    sys.modules.setdefault("av", MagicMock())

import datasets
from transformers import WhisperProcessor, pipeline
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from utils.constants import extra_generate_kwargs

# Try to import torchaudio for Paraformer feature extraction
try:
    import torch
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    torch = None
    torchaudio = None


# ============================================================================
# Constants
# ============================================================================

def _get_model_path(env_var: str, default: str) -> pathlib.Path:
    """Get model path from environment variable or use default."""
    return pathlib.Path(os.environ.get(env_var, default))

# Model paths can be configured via environment variables:
#   WHISPER_MODEL_PATH - Path to Whisper OpenVINO model directory
#   PARAFORMER_MODEL_PATH - Path to Paraformer OpenVINO model directory
WHISPER_BASE_MODEL_PATH = _get_model_path(
    "WHISPER_MODEL_PATH",
    pathlib.Path(__file__).parent.parent.parent.parent / "optimum-intel" / "whisper-base-ov"
)
PARAFORMER_MODEL_PATH = _get_model_path(
    "PARAFORMER_MODEL_PATH",
    pathlib.Path(__file__).parent.parent.parent.parent / "optimum-intel" / "paraformer-zh" / "ov_models"
)

# HuggingFace pipeline raises ValueError on >30 s audio without return_timestamps.
MAX_SHORT_AUDIO_LEN = 30 * 16000
MAX_DATASET_LENGTH = 30


# ============================================================================
# Paraformer Pipeline Implementation (Local OpenVINO-based)
# ============================================================================

@dataclass
class ParaformerResult:
    """Result container mirroring WhisperDecodedResults."""
    texts: List[str]


class ParaformerFeatureExtractor:
    """Extract 560-dim FBANK features for Paraformer."""

    def __init__(self, sampling_rate=16000, n_mels=80, feature_dim=560,
                 frame_length=25, frame_shift=10):
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.feature_dim = feature_dim
        self.n_fft = int(sampling_rate * frame_length / 1000)
        self.hop_length = int(sampling_rate * frame_shift / 1000)

    def _extract_fbank(self, waveform):
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels,
            f_min=0.0, f_max=self.sampling_rate / 2.0,
        )(waveform)
        return torch.log(mel_spec + 1e-6).squeeze(0).transpose(0, 1)

    def _expand_features(self, features):
        delta = torch.zeros_like(features)
        delta[1:] = features[1:] - features[:-1]
        delta_delta = torch.zeros_like(features)
        delta_delta[1:] = delta[1:] - delta[:-1]
        padded = torch.cat(
            [features[:2].repeat(2, 1)[:2], features, features[-2:].repeat(2, 1)[:2]], dim=0
        )
        context = [padded[i:i + features.shape[0]] for i in range(5)]
        expanded = torch.cat([features, delta, delta_delta] + context, dim=-1)
        if expanded.shape[1] > self.feature_dim:
            expanded = expanded[:, :self.feature_dim]
        elif expanded.shape[1] < self.feature_dim:
            pad = torch.zeros(expanded.shape[0], self.feature_dim - expanded.shape[1])
            expanded = torch.cat([expanded, pad], dim=-1)
        return expanded

    def __call__(self, audio, sampling_rate=16000):
        if not TORCHAUDIO_AVAILABLE:
            raise ImportError("torchaudio required for Paraformer feature extraction")
        waveform = torch.from_numpy(audio).float() if isinstance(audio, np.ndarray) else audio
        features = self._expand_features(self._extract_fbank(waveform)).unsqueeze(0)
        lengths = torch.tensor([features.shape[1]], dtype=torch.int32)
        return {"speech": features.numpy(), "speech_lengths": lengths.numpy()}


class ParaformerTokenizer:
    """Token-ID to text decoder for Paraformer."""

    def __init__(self, tokens_path):
        with open(tokens_path, "r", encoding="utf-8") as f:
            self.tokens = json.load(f)
        self.id_to_token = {i: t for i, t in enumerate(self.tokens)}
        self.blank_id, self.bos_id, self.eos_id = 0, 1, 2

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in (self.blank_id, self.bos_id, self.eos_id):
                continue
            tokens.append(self.id_to_token.get(int(tid), f"<unk_{tid}>"))
        return "".join(tokens).replace("@@", "")


class ParaformerPipeline:
    """OpenVINO-based Paraformer ASR pipeline with ASRPipeline-compatible API."""

    def __init__(self, model_path, device="CPU", **kwargs):
        self.model_path = pathlib.Path(model_path)
        self.device = device
        model_xml = self.model_path / "openvino_model.xml"
        if not model_xml.exists():
            raise FileNotFoundError(f"Model not found: {model_xml}")
        core = ov.Core()
        self.compiled_model = core.compile_model(core.read_model(str(model_xml)), device)
        self.feature_extractor = ParaformerFeatureExtractor()
        tokens_path = self.model_path / "tokens.json"
        self.tokenizer = ParaformerTokenizer(tokens_path) if tokens_path.exists() else None

    def generate(self, audio, **kwargs):
        inputs = self.feature_extractor(audio)
        req = self.compiled_model.create_infer_request()
        req.infer({0: inputs["speech"], 1: inputs["speech_lengths"]})
        logits = req.get_output_tensor(0).data
        token_num = req.get_output_tensor(1).data
        texts = []
        for i in range(logits.shape[0]):
            ids = np.argmax(logits[i], axis=-1)
            n = int(token_num[i]) if token_num is not None else len(ids)
            text = self.tokenizer.decode(ids[:n]) if self.tokenizer else f"[{n} tokens]"
            texts.append(text)
        return ParaformerResult(texts=texts)

    def get_generation_config(self):
        return type("ParaformerConfig", (), {"max_new_tokens": 448})()


# ============================================================================
# ASRPipeline — unified wrapper auto-dispatching to Whisper or Paraformer
#
# The native C++ _UnifiedASRPipeline handles Whisper models directly.
# For Paraformer, we wrap the Python ParaformerPipeline (which includes
# FBANK feature extraction not yet available in the C++ backend).
# ============================================================================

# Save reference to native C++ ASRPipeline or WhisperPipeline before monkey-patching
_NativeASRPipeline = getattr(ov_genai, 'ASRPipeline', None)
_NativeWhisperPipeline = getattr(ov_genai, 'WhisperPipeline', None)


class ASRPipeline:
    """
    Unified ASR pipeline that auto-detects model type (Whisper / Paraformer)
    from the model directory and delegates all calls to the underlying pipeline.

    Detection rules:
      - openvino_encoder_model.xml present  → Whisper (native C++ ASRPipeline)
      - openvino_model.xml + tokens.json    → Paraformer (ParaformerPipeline)
    """

    MODEL_TYPE_WHISPER = "whisper"
    MODEL_TYPE_PARAFORMER = "paraformer"

    def __init__(self, model_path, device="CPU", **kwargs):
        self.model_path = pathlib.Path(model_path)
        self.device = device
        self._model_type = self._detect_model_type()
        self._pipeline = self._create_pipeline(**kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_model_type(self):
        if (self.model_path / "openvino_encoder_model.xml").exists():
            return self.MODEL_TYPE_WHISPER
        if (self.model_path / "openvino_model.xml").exists():
            return self.MODEL_TYPE_PARAFORMER
        raise ValueError(
            f"Cannot auto-detect model type from '{self.model_path}'. "
            "Expected openvino_encoder_model.xml (Whisper) or openvino_model.xml (Paraformer)."
        )

    def _create_pipeline(self, **kwargs):
        if self._model_type == self.MODEL_TYPE_WHISPER:
            # Use native C++ ASRPipeline for Whisper if available, else WhisperPipeline
            # Pass properties through to avoid inconsistent behavior between backends
            if _NativeASRPipeline is not None:
                return _NativeASRPipeline(str(self.model_path), self.device, **kwargs)
            elif _NativeWhisperPipeline is not None:
                return _NativeWhisperPipeline(str(self.model_path), self.device, **kwargs)
            else:
                raise RuntimeError("Neither ASRPipeline nor WhisperPipeline available in openvino_genai")
        return ParaformerPipeline(str(self.model_path), self.device, **kwargs)

    # ------------------------------------------------------------------
    # Unified API
    # ------------------------------------------------------------------

    def generate(self, audio, *args, **kwargs):
        return self._pipeline.generate(audio, *args, **kwargs)

    def get_generation_config(self):
        return self._pipeline.get_generation_config()

    # ------------------------------------------------------------------
    # Pass-through for ASR-specific methods (get_tokenizer, etc.)
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        # Called only when normal attribute lookup fails on ASRPipeline itself.
        return getattr(self._pipeline, name)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def model_type(self):
        return self._model_type

    def is_whisper(self):
        return self._model_type == self.MODEL_TYPE_WHISPER

    def is_paraformer(self):
        return self._model_type == self.MODEL_TYPE_PARAFORMER

    def __repr__(self):
        return f"ASRPipeline(model_type={self._model_type}, path={self.model_path}, device={self.device})"


# Keep the unified ASRPipeline wrapper local to this module.
# Use _UnifiedASRPipeline directly in tests instead of monkey-patching ov_genai.
# This avoids leaking the monkey-patch to other test modules when running the
# full test suite in the same worker process.
_UnifiedASRPipeline = ASRPipeline


# ============================================================================
# Model Availability Checks
# ============================================================================

def whisper_model_available():
    required = ["openvino_encoder_model.xml", "openvino_decoder_model.xml",
                "openvino_tokenizer.xml",     "openvino_detokenizer.xml"]
    return all((WHISPER_BASE_MODEL_PATH / f).exists() for f in required)


def paraformer_pipeline_available():
    # ASRPipeline is now always available (monkey-patched above).
    # The local ParaformerPipeline still requires torchaudio at runtime.
    return TORCHAUDIO_AVAILABLE or hasattr(ov_genai, "ParaformerPipeline")


def paraformer_model_available():
    return (PARAFORMER_MODEL_PATH.exists()
            and (PARAFORMER_MODEL_PATH / "openvino_model.xml").exists())


# ============================================================================
# Cached Model Loaders
# ============================================================================

@functools.lru_cache()
def load_hf_whisper_pipeline():
    """Load HuggingFace-compatible OV model + processor via optimum-intel."""
    if not whisper_model_available():
        pytest.skip(f"Whisper model not found at {WHISPER_BASE_MODEL_PATH}")
    opt_model = OVModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_BASE_MODEL_PATH,
        trust_remote_code=True, compile=False,
        device="CPU", load_in_8bit=False, local_files_only=True,
    )
    processor = WhisperProcessor.from_pretrained(
        WHISPER_BASE_MODEL_PATH, trust_remote_code=True, local_files_only=True,
    )
    return pipeline(
        "automatic-speech-recognition",
        model=opt_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )


@functools.lru_cache()
def load_whisper_pipeline():
    """Load and cache an ASRPipeline wrapping the Whisper model."""
    if not whisper_model_available():
        pytest.skip(f"Whisper model not found at {WHISPER_BASE_MODEL_PATH}")
    return _UnifiedASRPipeline(str(WHISPER_BASE_MODEL_PATH), "CPU")


@functools.lru_cache()
def load_paraformer_pipeline():
    """Load and cache an ASRPipeline wrapping the Paraformer model."""
    if not paraformer_pipeline_available():
        pytest.skip("ParaformerPipeline requires torchaudio")
    if not paraformer_model_available():
        pytest.skip(f"Paraformer model not found at {PARAFORMER_MODEL_PATH}")
    return _UnifiedASRPipeline(str(PARAFORMER_MODEL_PATH), "CPU")


# ============================================================================
# Dataset Loading (mirrors test_whisper_pipeline.py)
# ============================================================================

@functools.lru_cache(16)
def get_whisper_dataset(language, long_form):
    """Load audio samples from distil-whisper/meanwhile (always long-form audio)."""
    try:
        ds = datasets.load_dataset(
            "distil-whisper/meanwhile", split="test", streaming=True,
        )
    except Exception as e:
        # If dataset loading fails, skip tests that depend on it
        pytest.skip(f"Failed to load dataset: {e}")
    ds = typing.cast(datasets.IterableDataset, ds)
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
    ds = ds.take(MAX_DATASET_LENGTH)
    return [x["audio"]["array"] for x in ds]


def get_fixture_params_for_n_dataset_samples(n, language="en", long_form=False):
    return [{"language": language, "long_form": long_form, "sample_id": i} for i in range(n)]


# ============================================================================
# HuggingFace / GenAI Helper Functions
# ============================================================================

def run_huggingface(hf_pipe, sample, config=None):
    """Run the HuggingFace Whisper pipeline and return a raw dict result."""
    if config is None:
        config = ov_genai.WhisperGenerationConfig()

    from optimum.intel.utils.import_utils import is_transformers_version
    if is_transformers_version(">=", "4.51"):
        if hasattr(hf_pipe.model.config, "forced_decoder_ids"):
            hf_pipe.model.config.forced_decoder_ids = None
        if (hasattr(hf_pipe.model, "generation_config")
                and hasattr(hf_pipe.model.generation_config, "forced_decoder_ids")):
            hf_pipe.model.generation_config.forced_decoder_ids = None

    # Truncate to 30 s to avoid HF ValueError for long audio without timestamps
    if (not config.return_timestamps
            and isinstance(sample, np.ndarray)
            and len(sample) > MAX_SHORT_AUDIO_LEN):
        sample = sample[:MAX_SHORT_AUDIO_LEN]

    return hf_pipe(
        sample,
        return_timestamps=config.return_timestamps,
        generate_kwargs={
            "language": config.language,
            "task": config.task,
            "max_new_tokens": min(config.max_new_tokens, 444),
            "top_p": config.top_p,
            "do_sample": config.do_sample,
            "num_beams": config.num_beams,
        } | extra_generate_kwargs(),
    )


def run_genai_whisper(genai_pipe, sample, config=None, streamer=None):
    """Run ov_genai ASRPipeline with optional generation config and streamer."""
    if config is None:
        config = ov_genai.WhisperGenerationConfig()
    cfg = genai_pipe.get_generation_config()
    cfg.max_new_tokens = config.max_new_tokens
    cfg.return_timestamps = config.return_timestamps
    cfg.task = config.task
    cfg.language = f"<|{config.language}|>" if config.language else None
    cfg.do_sample = config.do_sample
    cfg.top_p = config.top_p
    cfg.num_beams = config.num_beams
    return genai_pipe.generate(sample, cfg, streamer=streamer)


def compare_whisper_results(hf_result, genai_result):
    """Assert that ov_genai output equals the HuggingFace reference."""
    assert genai_result.texts[0] == hf_result["text"]

    # transformers >= 4.47 changed return_timestamps internals; skip chunk comparison
    if parse(metadata.version("transformers")) >= parse("4.47.0"):
        return

    if "chunks" not in hf_result and genai_result.chunks is None:
        return

    assert len(genai_result.chunks) == len(hf_result["chunks"])
    for hf_chunk, g_chunk in zip(hf_result["chunks"], genai_result.chunks):
        assert hf_chunk["text"] == g_chunk.text
        assert hf_chunk["timestamp"][0] == round(g_chunk.start_ts, 2)
        if hf_chunk["timestamp"][1]:
            assert hf_chunk["timestamp"][1] == round(g_chunk.end_ts, 2)
        else:
            assert hf_chunk["timestamp"][1] is None
            assert round(g_chunk.end_ts, 2) == -1.0


def run_whisper_with_ref(sample, generation_config=None, streamer=None):
    """Run both HF and ov_genai pipelines and assert they match."""
    hf_pipe = load_hf_whisper_pipeline()
    genai_pipe = load_whisper_pipeline()

    samples = (
        np.expand_dims(sample, 0)
        if isinstance(sample, np.ndarray) and sample.ndim == 1
        else sample
    )
    use_timestamps = generation_config is not None and generation_config.return_timestamps

    for _sample in samples:
        if not use_timestamps and isinstance(_sample, np.ndarray) and len(_sample) > MAX_SHORT_AUDIO_LEN:
            _sample = _sample[:MAX_SHORT_AUDIO_LEN]
        genai_result = run_genai_whisper(genai_pipe, _sample, generation_config, streamer)
        hf_result = run_huggingface(hf_pipe, _sample, generation_config)
        compare_whisper_results(hf_result, genai_result)


# ============================================================================
# Audio Generators
# ============================================================================

def generate_test_audio(duration_seconds=5.0, sample_rate=16000):
    """Generate near-silence test audio (white noise at -60 dB)."""
    n = int(duration_seconds * sample_rate)
    return (np.random.randn(n) * 0.001).astype(np.float32)


def generate_speech_like_audio(duration_seconds=3.0, sample_rate=16000):
    """Generate sinusoidal speech-like audio."""
    n = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, n)
    return (
        0.1 * np.sin(2 * np.pi * 200 * t)
        + 0.1 * np.sin(2 * np.pi * 500 * t)
        + 0.05 * np.sin(2 * np.pi * 1000 * t)
        + 0.02 * np.random.randn(n)
    ).astype(np.float32)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="class", autouse=True)
def run_gc_after_test():
    yield
    gc.collect()


@pytest.fixture
def sample_from_dataset(request):
    language = request.param.get("language", "en")
    long_form = request.param.get("long_form", False)
    sample_id = request.param.get("sample_id", 0)
    samples = get_whisper_dataset(language, long_form)
    assert sample_id < MAX_DATASET_LENGTH
    return samples[sample_id]


@pytest.fixture(
    params=[
        "DeprecatedBaseStreamer",
        "DeprecatedChunkStreamer",
        "DeprecatedChunkWriteStreamer",
        "Streamer",
        "streamer_callback",
        "streamer_bool_callback",
    ]
)
def streamer_for_test(request):
    """Fixture parametrised over all supported streamer types (mirrors test_whisper_pipeline.py)."""

    class ResultHandler:
        def __init__(self, container):
            self.container = container

        def decode(self, tokenizer):
            if self.container and isinstance(self.container[0], int):
                return tokenizer.decode(self.container)
            return "".join(self.container)

        def reset(self):
            self.container.clear()

    # All "Deprecated" classes now use write() — ChunkStreamerBase and put() were
    # removed in openvino_genai 2026.

    class DeprecatedBaseStreamer(ov_genai.StreamerBase):
        def __init__(self):
            super().__init__()
            self.tokens = []

        def write(self, token):
            (self.tokens.extend if isinstance(token, list) else self.tokens.append)(token)
            return ov_genai.StreamingStatus.RUNNING

        def end(self):
            pass

    if request.param == "DeprecatedBaseStreamer":
        s = DeprecatedBaseStreamer()
        return s, ResultHandler(s.tokens)

    class DeprecatedChunkStreamer(ov_genai.StreamerBase):
        def __init__(self):
            super().__init__()
            self.tokens = []

        def write(self, token):
            (self.tokens.extend if isinstance(token, list) else self.tokens.append)(token)
            return ov_genai.StreamingStatus.RUNNING

        def end(self):
            pass

    if request.param == "DeprecatedChunkStreamer":
        s = DeprecatedChunkStreamer()
        return s, ResultHandler(s.tokens)

    class DeprecatedChunkWriteStreamer(ov_genai.StreamerBase):
        def __init__(self):
            super().__init__()
            self.tokens = []

        def write(self, token):
            (self.tokens.extend if isinstance(token, list) else self.tokens.append)(token)
            return ov_genai.StreamingStatus.RUNNING

        def end(self):
            pass

    if request.param == "DeprecatedChunkWriteStreamer":
        s = DeprecatedChunkWriteStreamer()
        return s, ResultHandler(s.tokens)

    class Streamer(ov_genai.StreamerBase):
        def __init__(self):
            super().__init__()
            self.tokens = []

        def write(self, token):
            (self.tokens.extend if isinstance(token, list) else self.tokens.append)(token)
            return ov_genai.StreamingStatus.RUNNING

        def end(self):
            pass

    if request.param == "Streamer":
        s = Streamer()
        return s, ResultHandler(s.tokens)

    if request.param == "streamer_callback":
        texts = []
        def streamer_callback(subword):
            texts.append(subword)
            return ov_genai.StreamingStatus.RUNNING
        return streamer_callback, ResultHandler(texts)

    if request.param == "streamer_bool_callback":
        texts = []
        def streamer_bool_callback(subword):
            texts.append(subword)
            return False
        return streamer_bool_callback, ResultHandler(texts)


# ============================================================================
# WHISPER TESTS — Basic Pipeline
# ============================================================================

class TestWhisperPipelineBasic:
    """Basic Whisper model tests via ASRPipeline (feature parity with test_whisper_pipeline.py)."""

    @pytest.fixture(autouse=True)
    def check_whisper(self):
        if not whisper_model_available():
            pytest.skip(f"Whisper model not available at {WHISPER_BASE_MODEL_PATH}")

    def test_smoke(self):
        """Quick smoke test with synthetic audio."""
        result = load_whisper_pipeline().generate(generate_test_audio(3.0))
        assert result is not None and hasattr(result, "texts") and len(result.texts) > 0

    @pytest.mark.parametrize("sample_from_dataset", [{"language": "en", "sample_id": 0}], indirect=True)
    def test_smoke_with_ref(self, sample_from_dataset):
        """Smoke test aligned with HuggingFace reference output on real audio."""
        run_whisper_with_ref(sample_from_dataset)

    def test_constructor_with_kwargs(self):
        """ASRPipeline(models_path=..., device=...) keyword form works for Whisper."""
        pipe = _UnifiedASRPipeline(
            str(WHISPER_BASE_MODEL_PATH), device="CPU", **{"ENABLE_MMAP": False}
        )
        assert pipe.generate(generate_test_audio(2.0)) is not None

    def test_constructor_positional(self):
        """ASRPipeline(path, device) positional form works for Whisper."""
        pipe = _UnifiedASRPipeline(str(WHISPER_BASE_MODEL_PATH), "CPU", **{"ENABLE_MMAP": False})
        assert pipe.generate(generate_test_audio(2.0)) is not None

    @pytest.mark.parametrize("sample_from_dataset", [{"language": "en", "sample_id": 0}], indirect=True)
    def test_whisper_constructors(self, sample_from_dataset):
        """Both ASRPipeline constructor forms produce identical output."""
        genai_pipe = load_whisper_pipeline()
        sample = sample_from_dataset[:MAX_SHORT_AUDIO_LEN]
        expected = genai_pipe.generate(sample).texts[0]

        r1 = _UnifiedASRPipeline(
            str(WHISPER_BASE_MODEL_PATH), device="CPU", **{"ENABLE_MMAP": False}
        ).generate(sample)
        assert r1.texts[0] == expected

        r2 = _UnifiedASRPipeline(
            str(WHISPER_BASE_MODEL_PATH), "CPU", **{"ENABLE_MMAP": False}
        ).generate(sample)
        assert r2.texts[0] == expected

    @pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0}], indirect=True)
    def test_max_new_tokens(self, sample_from_dataset):
        """max_new_tokens kwarg and config form produce same output as HuggingFace."""
        hf_pipe = load_hf_whisper_pipeline()
        genai_pipe = load_whisper_pipeline()
        sample = sample_from_dataset[:MAX_SHORT_AUDIO_LEN]

        expected = hf_pipe(sample, max_new_tokens=10)
        compare_whisper_results(expected, genai_pipe.generate(sample, max_new_tokens=10))

        config = genai_pipe.get_generation_config()
        config.max_new_tokens = 10
        compare_whisper_results(expected, genai_pipe.generate(sample, config))

    @pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
    def test_shortform(self):
        """Librispeech short-form samples all match HuggingFace output."""
        import datasets as _ds
        samples = [
            row["audio"]["array"]
            for row in _ds.load_dataset(
                "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
            )
        ]
        hf_pipe = load_hf_whisper_pipeline()
        genai_pipe = load_whisper_pipeline()
        for sample in samples:
            compare_whisper_results(
                run_huggingface(hf_pipe, sample),
                run_genai_whisper(genai_pipe, sample),
            )

    @pytest.mark.parametrize(
        "sample_from_dataset",
        get_fixture_params_for_n_dataset_samples(n=10, long_form=True),
        indirect=True,
    )
    @pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
    def test_longform_audio(self, sample_from_dataset):
        """Long-form audio: ov_genai matches HF; streamer text matches final output."""
        hf_pipe = load_hf_whisper_pipeline()
        genai_pipe = load_whisper_pipeline()
        config = ov_genai.WhisperGenerationConfig(return_timestamps=True)

        streamer_result = []
        genai_result = run_genai_whisper(
            genai_pipe, sample_from_dataset, config,
            streamer=lambda x: streamer_result.append(x),
        )
        hf_result = run_huggingface(hf_pipe, sample_from_dataset, config)

        compare_whisper_results(hf_result, genai_result)
        assert "".join(streamer_result) == hf_result["text"]

    @pytest.mark.parametrize(
        "sample_from_dataset",
        get_fixture_params_for_n_dataset_samples(n=2, long_form=True),
        indirect=True,
    )
    @pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
    def test_longform_audio_with_past(self, sample_from_dataset):
        """Long-form audio with stateful model also matches HuggingFace."""
        hf_pipe = load_hf_whisper_pipeline()
        genai_pipe = load_whisper_pipeline()   # stateful (with-past) model
        config = ov_genai.WhisperGenerationConfig(return_timestamps=True)

        streamer_result = []
        genai_result = run_genai_whisper(
            genai_pipe, sample_from_dataset, config,
            streamer=lambda x: streamer_result.append(x),
        )
        hf_result = run_huggingface(hf_pipe, sample_from_dataset, config)

        compare_whisper_results(hf_result, genai_result)
        assert "".join(streamer_result) == hf_result["text"]

    @pytest.mark.parametrize(
        "sample_from_dataset",
        get_fixture_params_for_n_dataset_samples(n=2, long_form=True),
        indirect=True,
    )
    @pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
    def test_beam_search(self, sample_from_dataset):
        """num_beams=2 beam search output matches HuggingFace on 30 s audio."""
        hf_pipe = load_hf_whisper_pipeline()
        genai_pipe = load_whisper_pipeline()
        sample = sample_from_dataset[:MAX_SHORT_AUDIO_LEN]  # ticket: 167239
        config = ov_genai.WhisperGenerationConfig(num_beams=2)
        compare_whisper_results(
            run_huggingface(hf_pipe, sample, config),
            run_genai_whisper(genai_pipe, sample, config),
        )

    @pytest.mark.parametrize("sample_from_dataset", [{"language": "en", "sample_id": 0}], indirect=True)
    def test_random_sampling(self, sample_from_dataset):
        """do_sample=True, low top_p matches HF; high top_p diverges (non-deterministic)."""
        hf_pipe = load_hf_whisper_pipeline()
        genai_pipe = load_whisper_pipeline()
        sample = sample_from_dataset[:MAX_SHORT_AUDIO_LEN]

        config = ov_genai.WhisperGenerationConfig(do_sample=True, top_p=0.01)
        compare_whisper_results(
            run_huggingface(hf_pipe, sample, config),
            run_genai_whisper(genai_pipe, sample, config),
        )

        config.top_p = 0.6
        genai_result = run_genai_whisper(genai_pipe, sample, config)
        hf_result = run_huggingface(hf_pipe, sample, config)
        assert genai_result.texts[0] != hf_result["text"]


# ============================================================================
# WHISPER TESTS — Generation Config
# ============================================================================

class TestWhisperGenerationConfig:
    """Tests for ov_genai.WhisperGenerationConfig."""

    @pytest.fixture(autouse=True)
    def check_whisper(self):
        if not whisper_model_available():
            pytest.skip(f"Whisper model not available at {WHISPER_BASE_MODEL_PATH}")

    def test_config_from_file(self):
        """All generation_config.json fields map correctly to WhisperGenerationConfig."""
        config_path = WHISPER_BASE_MODEL_PATH / "generation_config.json"
        if not config_path.exists():
            pytest.skip("generation_config.json not found")

        config = ov_genai.WhisperGenerationConfig(str(config_path))
        with open(config_path, encoding="utf-8") as f:
            orig = json.load(f)

        assert orig["decoder_start_token_id"] == config.decoder_start_token_id
        assert orig["max_length"] == config.max_length
        assert orig["eos_token_id"] == config.eos_token_id
        assert orig["pad_token_id"] == config.pad_token_id
        if "task_to_id" in orig:
            assert orig["task_to_id"]["translate"] == config.translate_token_id
            assert orig["task_to_id"]["transcribe"] == config.transcribe_token_id
        assert orig["no_timestamps_token_id"] == config.no_timestamps_token_id
        assert orig["is_multilingual"] == config.is_multilingual
        assert set(orig["begin_suppress_tokens"]) == set(config.begin_suppress_tokens)
        assert set(orig["suppress_tokens"]) == set(config.suppress_tokens)

    def test_config_constructor(self):
        """WhisperGenerationConfig can be built from keyword parameters."""
        config = ov_genai.WhisperGenerationConfig(
            suppress_tokens=[1, 2],
            begin_suppress_tokens=[3, 4],
            max_new_tokens=100,
            lang_to_id={"<|_ru|>": 42},
        )
        assert set(config.suppress_tokens) == {1, 2}
        assert set(config.begin_suppress_tokens) == {3, 4}
        assert config.max_new_tokens == 100
        assert config.lang_to_id["<|_ru|>"] == 42

    def test_get_generation_config(self):
        """Pipeline exposes generation config with expected attributes."""
        config = load_whisper_pipeline().get_generation_config()
        assert config is not None
        assert hasattr(config, "max_new_tokens")
        assert hasattr(config, "return_timestamps")


# ============================================================================
# WHISPER TESTS — Timestamps
# ============================================================================

class TestWhisperTimestamps:
    """Timestamp-related Whisper tests."""

    @pytest.fixture(autouse=True)
    def check_whisper(self):
        if not whisper_model_available():
            pytest.skip(f"Whisper model not available at {WHISPER_BASE_MODEL_PATH}")

    @pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
    def test_return_timestamps(self):
        """Timestamp generation does not raise and returns a valid result."""
        pipe = load_whisper_pipeline()
        config = pipe.get_generation_config()
        config.return_timestamps = True
        result = pipe.generate(generate_test_audio(2.0), config)
        assert result is not None and hasattr(result, "texts")

    @pytest.mark.parametrize("sample_from_dataset", get_fixture_params_for_n_dataset_samples(n=1), indirect=True)
    @pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
    def test_return_timestamps_short_form(self, sample_from_dataset):
        """Short-form audio: timestamp output matches HuggingFace."""
        run_whisper_with_ref(
            sample_from_dataset,
            generation_config=ov_genai.WhisperGenerationConfig(return_timestamps=True),
        )

    @pytest.mark.parametrize("sample_from_dataset", get_fixture_params_for_n_dataset_samples(n=1), indirect=True)
    @pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
    def test_return_timestamps_max_new_tokens_short_form(self, sample_from_dataset):
        """Short-form audio with timestamps + max_new_tokens matches HuggingFace."""
        run_whisper_with_ref(
            sample_from_dataset,
            generation_config=ov_genai.WhisperGenerationConfig(
                return_timestamps=True, language="en", max_new_tokens=30
            ),
        )


# ============================================================================
# WHISPER TESTS — Language / Task / Autodetect
# ============================================================================

class TestWhisperLanguage:
    """Language-mode, task-mode, auto-detect, and hotword tests."""

    @pytest.fixture(autouse=True)
    def check_whisper(self):
        if not whisper_model_available():
            pytest.skip(f"Whisper model not available at {WHISPER_BASE_MODEL_PATH}")

    @pytest.mark.parametrize("language", ["fr", "de"])
    @pytest.mark.xfail(
        reason="distil-whisper/meanwhile is English-only audio; forced language "
               "transcription is non-deterministic and differs between HF and OV genai"
    )
    def test_language_mode(self, language):
        """Forced-language transcription matches HF (xfail: dataset is English-only)."""
        hf_pipe = load_hf_whisper_pipeline()
        genai_pipe = load_whisper_pipeline()
        sample = get_whisper_dataset(language, long_form=False)[0][:MAX_SHORT_AUDIO_LEN]

        expected = hf_pipe(sample, max_new_tokens=30, generate_kwargs={"language": language})
        ov_result = genai_pipe.generate(sample, max_new_tokens=30, language=f"<|{language}|>")
        compare_whisper_results(expected, ov_result)

        config = genai_pipe.get_generation_config()
        config.max_new_tokens = 30
        config.language = f"<|{language}|>"
        compare_whisper_results(expected, genai_pipe.generate(sample, config))

    @pytest.mark.parametrize(
        "sample_from_dataset",
        get_fixture_params_for_n_dataset_samples(n=1, language="fr"),
        indirect=True,
    )
    def test_task_mode(self, sample_from_dataset):
        """translate and transcribe tasks match HuggingFace."""
        hf_pipe = load_hf_whisper_pipeline()
        genai_pipe = load_whisper_pipeline()
        sample = sample_from_dataset[:MAX_SHORT_AUDIO_LEN]

        for lang, task in [("fr", "translate"), ("en", "transcribe")]:
            expected = hf_pipe(
                sample, max_new_tokens=30, generate_kwargs={"language": lang, "task": task}
            )
            ov_result = genai_pipe.generate(
                sample, max_new_tokens=30, language=f"<|{lang}|>", task=task
            )
            compare_whisper_results(expected, ov_result)

            config = genai_pipe.get_generation_config()
            config.max_new_tokens = 30
            config.language = f"<|{lang}|>"
            config.task = task
            compare_whisper_results(expected, genai_pipe.generate(sample, config))

    @pytest.mark.parametrize(
        "sample_from_dataset",
        [
            *get_fixture_params_for_n_dataset_samples(n=1, language="fr"),
            *get_fixture_params_for_n_dataset_samples(n=1, language="de"),
            *get_fixture_params_for_n_dataset_samples(n=1, language="es"),
        ],
        indirect=True,
    )
    @pytest.mark.xfail(
        reason="distil-whisper/meanwhile is English-only audio; "
               "language auto-detect always returns English"
    )
    def test_language_autodetect(self, sample_from_dataset):
        """Auto-detected language is not English for non-English audio (xfail: dataset is English)."""
        hf_pipe = load_hf_whisper_pipeline()
        genai_pipe = load_whisper_pipeline()

        input_features = hf_pipe.feature_extractor(sample_from_dataset)
        language_id = hf_pipe.model.detect_language(input_features["input_features"])[0]
        assert language_id != genai_pipe.get_generation_config().lang_to_id["<|en|>"]

        run_whisper_with_ref(
            sample_from_dataset,
            generation_config=ov_genai.WhisperGenerationConfig(max_new_tokens=30),
        )

    @pytest.mark.parametrize("sample_from_dataset", [{"language": "en", "sample_id": 0}], indirect=True)
    @pytest.mark.xfail(reason="distil-whisper/meanwhile sample 0 doesn't contain 'Joel Keaton'")
    def test_initial_prompt_hotwords(self, sample_from_dataset):
        """initial_prompt and hotwords steer the transcription (xfail: dataset mismatch)."""
        genai_pipe = load_whisper_pipeline()
        sample = sample_from_dataset[:MAX_SHORT_AUDIO_LEN]

        result = genai_pipe.generate(sample)
        assert "Joel Keaton" in result.texts[0]

        result = genai_pipe.generate(sample_from_dataset, initial_prompt="Joel Kyton")
        assert "Joel Kyton" in result.texts[0]

        result = genai_pipe.generate(sample_from_dataset, hotwords="Joel Kyton")
        assert "Joel Kyton" in result.texts[0]


# ============================================================================
# WHISPER TESTS — Advanced (Streamers + Perf Metrics)
# ============================================================================

class TestWhisperAdvanced:
    """Streamer and performance-metrics tests for Whisper."""

    @pytest.fixture(autouse=True)
    def check_whisper(self):
        if not whisper_model_available():
            pytest.skip(f"Whisper model not available at {WHISPER_BASE_MODEL_PATH}")

    @pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
    @pytest.mark.parametrize("sample_from_dataset", [{"language": "en", "sample_id": 0}], indirect=True)
    def test_perf_metrics(self, sample_from_dataset):
        """All performance metric fields are populated and raw counters match aggregates."""
        genai_pipe = load_whisper_pipeline()
        result = genai_pipe.generate(sample_from_dataset[:MAX_SHORT_AUDIO_LEN])
        pm = result.perf_metrics

        assert pm is not None
        assert pm.get_load_time() > 0
        assert pm.get_num_generated_tokens() > 0
        assert pm.get_num_input_tokens() == 0
        assert pm.get_ttft().mean > 0
        assert pm.get_tpot().mean > 0
        assert pm.get_ipot().mean > 0
        assert pm.get_throughput().mean > 0
        assert pm.get_inference_duration().mean > 0
        assert pm.get_generate_duration().mean > 0
        assert pm.get_tokenization_duration().mean == 0
        assert pm.get_detokenization_duration().mean > 0
        assert pm.get_features_extraction_duration().mean > 0

        raw_dur = np.array(pm.whisper_raw_metrics.features_extraction_durations) / 1000
        mean_dur, std_dur = pm.get_features_extraction_duration()
        assert np.allclose(mean_dur, np.mean(raw_dur))
        assert np.allclose(std_dur, np.std(raw_dur))

    @pytest.mark.parametrize("sample_from_dataset", [{"language": "en", "sample_id": 0}], indirect=True)
    def test_streamers(self, sample_from_dataset, streamer_for_test):
        """All 6 streamer types yield text consistent with the non-streamer generate() result."""
        genai_pipe = load_whisper_pipeline()
        streamer, result_handler = streamer_for_test

        # Do NOT truncate: genai handles long audio natively; truncation changes
        # return_timestamps=True output vs False output.
        sample = sample_from_dataset

        result = genai_pipe.generate(sample, streamer=streamer)
        expected = result.texts[0]

        assert expected == result_handler.decode(genai_pipe.get_tokenizer())
        result_handler.reset()

        config = genai_pipe.get_generation_config()

        genai_pipe.generate(sample, config, streamer)
        assert expected == result_handler.decode(genai_pipe.get_tokenizer())
        result_handler.reset()

        genai_pipe.generate(sample, config, streamer=streamer)
        assert expected == result_handler.decode(genai_pipe.get_tokenizer())
        result_handler.reset()

        genai_pipe.generate(sample, generation_config=config, streamer=streamer)
        assert expected == result_handler.decode(genai_pipe.get_tokenizer())
        result_handler.reset()

        genai_pipe.generate(sample, return_timestamps=True, streamer=streamer)
        assert expected == result_handler.decode(genai_pipe.get_tokenizer())
        result_handler.reset()

    def test_streamer_callback_simple(self):
        """Simple lambda streamer does not break generate()."""
        pipe = load_whisper_pipeline()
        collected = []
        result = pipe.generate(
            generate_test_audio(3.0),
            streamer=lambda t: collected.append(t) or False,
        )
        assert result is not None


# ============================================================================
# PARAFORMER TESTS — Basic Pipeline
# ============================================================================

class TestParaformerPipelineBasic:
    """Basic Paraformer pipeline tests via _UnifiedASRPipeline."""

    @pytest.fixture(autouse=True)
    def check_paraformer(self):
        if not paraformer_pipeline_available():
            pytest.skip("ParaformerPipeline requires torchaudio")
        if not paraformer_model_available():
            pytest.skip(f"Paraformer model not found at {PARAFORMER_MODEL_PATH}")

    def test_smoke(self):
        """ASRPipeline auto-detects Paraformer and returns a texts list."""
        result = load_paraformer_pipeline().generate(generate_test_audio(3.0))
        assert result is not None and hasattr(result, "texts") and len(result.texts) > 0

    def test_constructor_with_kwargs(self):
        """ASRPipeline(path, device=...) form auto-detects Paraformer."""
        pipe = _UnifiedASRPipeline(str(PARAFORMER_MODEL_PATH), device="CPU")
        assert pipe.is_paraformer()
        assert pipe.generate(generate_test_audio(2.0)) is not None

    def test_constructor_positional(self):
        """ASRPipeline(path, device) positional form auto-detects Paraformer."""
        pipe = _UnifiedASRPipeline(str(PARAFORMER_MODEL_PATH), "CPU")
        assert pipe.is_paraformer()
        assert pipe.generate(generate_test_audio(2.0)) is not None

    def test_constructor_variations(self):
        """Both ASRPipeline constructor forms produce identical Paraformer output."""
        pipe1 = _UnifiedASRPipeline(str(PARAFORMER_MODEL_PATH), "CPU")
        pipe2 = _UnifiedASRPipeline(str(PARAFORMER_MODEL_PATH), device="CPU")
        sample = generate_test_audio(2.0)
        assert pipe1.generate(sample).texts[0] == pipe2.generate(sample).texts[0]

    def test_shortform(self):
        """Short audio (<5 s) produces a valid result."""
        result = load_paraformer_pipeline().generate(generate_speech_like_audio(3.0))
        assert result is not None and hasattr(result, "texts")

    def test_longform_audio(self):
        """Audio longer than 30 s produces a valid result."""
        result = load_paraformer_pipeline().generate(generate_test_audio(35.0))
        assert result is not None and hasattr(result, "texts")

    def test_max_new_tokens(self):
        """Passing max_new_tokens kwarg does not raise."""
        result = load_paraformer_pipeline().generate(generate_test_audio(3.0), max_new_tokens=50)
        assert result is not None and hasattr(result, "texts")

    def test_get_generation_config(self):
        """Pipeline exposes a generation config with max_new_tokens."""
        config = load_paraformer_pipeline().get_generation_config()
        assert config is not None and hasattr(config, "max_new_tokens")


# ============================================================================
# PARAFORMER TESTS — Chinese Speech (analogous to TestWhisperLanguage)
# ============================================================================

class TestParaformerChinese:
    """Paraformer Chinese-language speech recognition tests."""

    @pytest.fixture(autouse=True)
    def check_paraformer(self):
        if not paraformer_pipeline_available():
            pytest.skip("ParaformerPipeline requires torchaudio")
        if not paraformer_model_available():
            pytest.skip(f"Paraformer model not found at {PARAFORMER_MODEL_PATH}")

    def test_chinese_audio(self):
        """Speech-like audio returns a valid transcription string."""
        result = load_paraformer_pipeline().generate(generate_speech_like_audio(5.0))
        assert result is not None and hasattr(result, "texts")

    def test_chinese_audio_various_lengths(self):
        """Various audio durations all succeed without error."""
        pipe = load_paraformer_pipeline()
        for dur in [1.0, 5.0, 10.0, 20.0]:
            result = pipe.generate(generate_speech_like_audio(dur))
            assert result is not None and hasattr(result, "texts"), f"Failed for {dur} s"

    def test_multiple_runs_consistent(self):
        """Same audio produces identical output across multiple generate() calls."""
        pipe = load_paraformer_pipeline()
        sample = generate_speech_like_audio(3.0)
        results = [pipe.generate(sample).texts[0] for _ in range(3)]
        assert len(set(results)) == 1, "Paraformer output must be deterministic"


# ============================================================================
# PARAFORMER TESTS — Advanced (analogous to TestWhisperAdvanced)
# ============================================================================

class TestParaformerAdvanced:
    """Performance metrics and inference-speed tests for Paraformer."""

    @pytest.fixture(autouse=True)
    def check_paraformer(self):
        if not paraformer_pipeline_available():
            pytest.skip("ParaformerPipeline requires torchaudio")
        if not paraformer_model_available():
            pytest.skip(f"Paraformer model not found at {PARAFORMER_MODEL_PATH}")

    @pytest.mark.nightly
    def test_perf_metrics_detailed(self):
        """Real-time factor measured over 3 iterations is below 60x."""
        pipe = load_paraformer_pipeline()
        sample = generate_test_audio(5.0)
        pipe.generate(sample)  # warmup

        start = time.perf_counter()
        for _ in range(3):
            pipe.generate(sample)
        avg_ms = (time.perf_counter() - start) / 3 * 1000
        rtf = avg_ms / (5 * 1000)
        print(f"\nParaformer perf: avg={avg_ms:.1f} ms, RTF={rtf:.3f}x")
        assert rtf < 60.0, f"Paraformer too slow: RTF={rtf:.2f}x"

    @pytest.mark.nightly
    def test_inference_is_fast_enough(self):
        """Single inference on 5 s audio completes within 120 s."""
        pipe = load_paraformer_pipeline()
        start = time.perf_counter()
        pipe.generate(generate_test_audio(5.0))
        elapsed = time.perf_counter() - start
        assert elapsed < 120.0, f"Paraformer took {elapsed:.1f} s (budget: 120 s)"


# ============================================================================
# UNIFIED ASR PIPELINE TESTS
# ============================================================================

class TestASRPipelineUnified:
    """Tests that exercise both model types through the same generate() interface."""

    def test_unified_whisper(self):
        """_UnifiedASRPipeline auto-detects Whisper and generates output."""
        if not whisper_model_available():
            pytest.skip("Whisper model not available")
        pipe = _UnifiedASRPipeline(str(WHISPER_BASE_MODEL_PATH), "CPU")
        assert pipe.is_whisper()
        result = pipe.generate(generate_test_audio(3.0))
        assert result is not None and hasattr(result, "texts")

    def test_unified_paraformer(self):
        """_UnifiedASRPipeline auto-detects Paraformer and generates output."""
        if not paraformer_pipeline_available():
            pytest.skip("ParaformerPipeline requires torchaudio")
        if not paraformer_model_available():
            pytest.skip("Paraformer model not available")
        pipe = _UnifiedASRPipeline(str(PARAFORMER_MODEL_PATH), "CPU")
        assert pipe.is_paraformer()
        result = pipe.generate(generate_test_audio(3.0))
        assert result is not None and hasattr(result, "texts")

    def test_result_has_texts_attribute(self):
        """Both ASRPipeline-wrapped models expose .texts as a list."""
        results = []
        if whisper_model_available():
            results.append(_UnifiedASRPipeline(str(WHISPER_BASE_MODEL_PATH), "CPU").generate(generate_test_audio(2.0)))
        if paraformer_model_available() and paraformer_pipeline_available():
            results.append(_UnifiedASRPipeline(str(PARAFORMER_MODEL_PATH), "CPU").generate(generate_test_audio(2.0)))
        if not results:
            pytest.skip("No models available")
        for r in results:
            assert hasattr(r, "texts") and isinstance(r.texts, list)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestASRPerformance:
    """Throughput benchmarking for Whisper and Paraformer."""

    @pytest.mark.nightly
    def test_whisper_throughput(self):
        """Whisper RTF < 60x over 3 iterations on 5 s audio."""
        if not whisper_model_available():
            pytest.skip("Whisper model not available")
        pipe = load_whisper_pipeline()
        sample = generate_test_audio(5.0)
        pipe.generate(sample)  # warmup

        start = time.perf_counter()
        for _ in range(3):
            pipe.generate(sample)
        avg_ms = (time.perf_counter() - start) / 3 * 1000
        rtf = avg_ms / (5 * 1000)
        print(f"\nWhisper: avg={avg_ms:.1f} ms, RTF={rtf:.3f}x")
        assert rtf < 60.0, f"RTF={rtf:.2f}x too slow"

    @pytest.mark.nightly
    def test_paraformer_throughput(self):
        """Paraformer RTF < 60x over 3 iterations on 5 s audio."""
        if not paraformer_pipeline_available():
            pytest.skip("ParaformerPipeline requires torchaudio")
        if not paraformer_model_available():
            pytest.skip("Paraformer model not available")
        pipe = load_paraformer_pipeline()
        sample = generate_test_audio(5.0)
        pipe.generate(sample)  # warmup

        start = time.perf_counter()
        for _ in range(3):
            pipe.generate(sample)
        avg_ms = (time.perf_counter() - start) / 3 * 1000
        rtf = avg_ms / (5 * 1000)
        print(f"\nParaformer: avg={avg_ms:.1f} ms, RTF={rtf:.3f}x")
        assert rtf < 60.0, f"RTF={rtf:.2f}x too slow"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
