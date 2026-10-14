# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import math
from pathlib import Path

import numpy as np
import openvino
import pytest

from openvino_genai import GenerationConfig, OmniPipeline, OmniTalkerSpeechConfig, StreamingStatus
from utils.atomic_download import AtomicDownloadManager
from utils.constants import get_ov_cache_converted_models_dir
from utils.hugging_face import export_with_optimum_cli, sanitize_model_id

MODEL_ID = "optimum-intel-internal-testing/tiny-random-qwen3-omni"
PROMPT = "Say hello."
# Bounds the talker so a chunk size larger than the whole utterance is easy to pick.
TALKER_MAX_NEW_TOKENS = 12


@pytest.fixture(scope="module")
def model_path() -> Path:
    path = get_ov_cache_converted_models_dir() / f"{sanitize_model_id(MODEL_ID)}_image-text-to-text"
    AtomicDownloadManager(path).execute(
        lambda temp_path: export_with_optimum_cli(MODEL_ID, "image-text-to-text", temp_path, trust_remote_code=True)
    )
    return path


@pytest.fixture(scope="module")
def samples_per_frame(model_path: Path) -> int:
    code2wav = json.loads((model_path / "config.json").read_text(encoding="utf-8"))["code2wav_config"]
    return math.prod(code2wav["upsample_rates"] + code2wav["upsampling_ratios"])


@pytest.fixture(scope="module")
def pipe(model_path: Path) -> OmniPipeline:
    return OmniPipeline(model_path, "CPU")


def speech_config(audio_chunk_frames: int) -> OmniTalkerSpeechConfig:
    config = OmniTalkerSpeechConfig()
    config.max_new_tokens = TALKER_MAX_NEW_TOKENS
    config.audio_chunk_frames = audio_chunk_frames
    return config


def generate(pipe: OmniPipeline, audio_chunk_frames: int, chunks: list[np.ndarray] | None = None) -> np.ndarray:
    def collect(chunk: openvino.Tensor) -> StreamingStatus:
        chunks.append(chunk.data.reshape(-1).copy())
        return StreamingStatus.RUNNING

    result = pipe.generate(
        PROMPT,
        text_config=GenerationConfig(max_new_tokens=8),
        talker_speech_config=speech_config(audio_chunk_frames),
        speech_streamer=collect if chunks is not None else None,
    )
    return result.speech_result.waveforms[0].data.reshape(-1)


@pytest.fixture(scope="module")
def full_waveform(pipe: OmniPipeline) -> np.ndarray:
    return generate(pipe, audio_chunk_frames=1)


@pytest.mark.vlm
@pytest.mark.parametrize("audio_chunk_frames", [1, 4, TALKER_MAX_NEW_TOKENS + 1])
def test_streamed_audio_matches_full_decode_length(
    pipe: OmniPipeline, full_waveform: np.ndarray, samples_per_frame: int, audio_chunk_frames: int
) -> None:
    chunks: list[np.ndarray] = []
    streamed_waveform = generate(pipe, audio_chunk_frames, chunks)

    assert chunks, "The speech streamer received no chunks"
    assert sum(chunk.size for chunk in chunks) == full_waveform.size
    np.testing.assert_array_equal(streamed_waveform, np.concatenate(chunks))
    # The code2wav warmup is trimmed once per utterance, so it shortens only the first chunk.
    for chunk in chunks[1:-1]:
        assert chunk.size == audio_chunk_frames * samples_per_frame
    if len(chunks) > 1:
        assert chunks[-1].size % samples_per_frame == 0
