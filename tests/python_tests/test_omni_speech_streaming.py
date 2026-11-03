# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import math
from pathlib import Path

import numpy as np
import openvino
import pytest

from openvino_genai import GenerationConfig, OmniPipeline, OmniTalkerSpeechConfig, StreamingStatus

PROMPT = "Say hello."
# Bounds the talker so a chunk size larger than the whole utterance is easy to pick.
TALKER_MAX_NEW_TOKENS = 12
DEFAULT_LEFT_CONTEXT_FRAMES = OmniTalkerSpeechConfig().audio_chunk_left_context_frames
# Chunks are decoded from different frame windows than the full decode, so samples match up to float noise only.
MATCH_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def samples_per_frame(omni_model_path: Path) -> int:
    code2wav = json.loads((omni_model_path / "config.json").read_text(encoding="utf-8"))["code2wav_config"]
    return math.prod(code2wav["upsample_rates"] + code2wav["upsampling_ratios"])


@pytest.fixture(scope="module")
def pipe(omni_model_path: Path) -> OmniPipeline:
    return OmniPipeline(omni_model_path, "CPU")


def speech_config(audio_chunk_frames: int, left_context_frames: int) -> OmniTalkerSpeechConfig:
    config = OmniTalkerSpeechConfig()
    config.max_new_tokens = TALKER_MAX_NEW_TOKENS
    config.audio_chunk_frames = audio_chunk_frames
    config.audio_chunk_left_context_frames = left_context_frames
    return config


def generate(
    pipe: OmniPipeline,
    audio_chunk_frames: int,
    chunks: list[np.ndarray] | None = None,
    left_context_frames: int = DEFAULT_LEFT_CONTEXT_FRAMES,
) -> np.ndarray:
    def collect(chunk: openvino.Tensor) -> StreamingStatus:
        chunks.append(chunk.data.reshape(-1).copy())
        return StreamingStatus.RUNNING

    result = pipe.generate(
        PROMPT,
        text_config=GenerationConfig(max_new_tokens=8),
        talker_speech_config=speech_config(audio_chunk_frames, left_context_frames),
        speech_streamer=collect if chunks is not None else None,
    )
    return result.speech_result.waveforms[0].data.reshape(-1)


def stream(pipe: OmniPipeline, audio_chunk_frames: int, left_context_frames: int) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    generate(pipe, audio_chunk_frames, chunks, left_context_frames)
    assert chunks, "The speech streamer received no chunks"
    return chunks


def max_error(streamed: np.ndarray, full: np.ndarray) -> float:
    return float(np.abs(streamed - full).max() / np.abs(full).max())


@pytest.fixture(scope="module")
def full_waveform(pipe: OmniPipeline) -> np.ndarray:
    return generate(pipe, audio_chunk_frames=1)


@pytest.mark.vlm
@pytest.mark.parametrize("audio_chunk_frames", [1, 4, TALKER_MAX_NEW_TOKENS + 1])
def test_streamed_audio_matches_full_decode(
    pipe: OmniPipeline, full_waveform: np.ndarray, samples_per_frame: int, audio_chunk_frames: int
) -> None:
    chunks: list[np.ndarray] = []
    streamed_waveform = generate(pipe, audio_chunk_frames, chunks)

    assert chunks, "The speech streamer received no chunks"
    np.testing.assert_array_equal(streamed_waveform, np.concatenate(chunks))
    # The code2wav warmup is trimmed once per utterance, so it shortens only the first chunk.
    for chunk in chunks[1:-1]:
        assert chunk.size == audio_chunk_frames * samples_per_frame
    if len(chunks) > 1:
        assert chunks[-1].size % samples_per_frame == 0
    # The default context covers the whole utterance, so every chunk sees the history the full decode sees.
    assert TALKER_MAX_NEW_TOKENS <= DEFAULT_LEFT_CONTEXT_FRAMES
    assert max_error(streamed_waveform, full_waveform) < MATCH_TOLERANCE


@pytest.mark.vlm
def test_zero_left_context_trims_warmup_from_every_chunk(pipe: OmniPipeline, full_waveform: np.ndarray) -> None:
    chunks = stream(pipe, audio_chunk_frames=2, left_context_frames=0)

    assert len(chunks) > 1
    for chunk in chunks[1:-1]:
        assert chunk.size == chunks[0].size
    assert sum(chunk.size for chunk in chunks) < full_waveform.size


@pytest.mark.vlm
def test_more_left_context_brings_streamed_audio_closer_to_full_decode(
    pipe: OmniPipeline, full_waveform: np.ndarray
) -> None:
    errors = []
    for left_context_frames in (1, 4, DEFAULT_LEFT_CONTEXT_FRAMES):
        chunks = stream(pipe, audio_chunk_frames=2, left_context_frames=left_context_frames)
        errors.append(max_error(np.concatenate(chunks), full_waveform))

    assert errors == sorted(errors, reverse=True) and len(set(errors)) == len(errors), errors
    assert errors[0] > MATCH_TOLERANCE, errors
