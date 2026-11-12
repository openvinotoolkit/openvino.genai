# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import datasets
import openvino_genai as ov_genai
import pytest

from utils.dataset_utils import load_parquet_dataset_via_snapshot


@pytest.fixture(scope="module")
def voxtral_pipeline():
    model_dir = os.environ.get("VOXTRAL_OV_MODEL")
    if not model_dir:
        pytest.skip("Set VOXTRAL_OV_MODEL to an exported Voxtral Realtime model directory.")
    assert Path(model_dir).is_dir(), f"Voxtral model directory does not exist: {model_dir}"
    return ov_genai.ASRPipeline(model_dir, "CPU", NUM_STREAMS=2, INFERENCE_NUM_THREADS=4)


@pytest.fixture(scope="module")
def voxtral_audio_inputs(voxtral_pipeline):
    dataset = load_parquet_dataset_via_snapshot(
        "hf-internal-testing/librispeech_asr_dummy",
        {"validation": "clean/validation-00000-of-00001.parquet"},
        revision="5be91486e11a2d616f4ec5db8d3fd248585ac07a",
        split="validation",
    )
    dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    return [dataset[index]["audio"]["array"][: 16000 * (4 + index)] for index in range(2)]


def _assert_stage_durations(stream, result):
    durations = stream.get_stage_durations()
    stages = {"audio_inference", "text_inference", "audio_kv_cache", "text_kv_cache"}
    assert stages <= durations.keys()
    assert all(durations[stage] > 0 for stage in stages)
    assert all(duration >= 0 for duration in durations.values())
    raw_metrics = result.perf_metrics.asr_raw_metrics
    assert durations["audio_inference"] == pytest.approx(sum(raw_metrics.encode_inference_durations) / 1000)
    assert durations["text_inference"] == pytest.approx(sum(raw_metrics.decode_inference_durations) / 1000)
    assert sum(durations[stage] for stage in stages) <= result.perf_metrics.get_generate_duration().mean


def _transcribe(stream, audio, frame_samples, first_delta_barrier=None):
    deltas = []
    cache_sizes = []
    assert stream.get_kv_cache_size() == 0
    last_stage_durations = stream.get_stage_durations()
    assert all(duration == 0 for duration in last_stage_durations.values())

    def on_text(delta):
        nonlocal last_stage_durations
        deltas.append(delta)
        cache_sizes.append(stream.get_kv_cache_size())
        windows = stream.get_kv_cache_window_sizes()
        assert set(windows) == {"audio", "text"}
        assert all(0 < used <= maximum for used, maximum in windows.values())
        stage_durations = stream.get_stage_durations()
        assert all(
            stage_durations[stage] >= last_stage_durations[stage]
            for stage in ("audio_inference", "text_inference", "audio_kv_cache", "text_kv_cache")
        )
        assert all(duration > 0 for duration in stage_durations.values())
        last_stage_durations = stage_durations
        if len(deltas) == 1 and first_delta_barrier is not None:
            # Neither session can finish before both have advanced their inference state.
            first_delta_barrier.wait(timeout=120)
        return ov_genai.StreamingStatus.RUNNING

    with ThreadPoolExecutor(max_workers=1) as executor:
        generation = executor.submit(stream.generate, on_text)
        for start in range(0, len(audio), frame_samples):
            stream.push_audio(audio[start : start + frame_samples].tolist())
        stream.end_audio()
        result = generation.result()

    assert len(result.texts) == 1
    assert deltas, "Expected at least one transcript delta."
    assert "".join(deltas) == result.texts[0]
    assert cache_sizes[0] > 0
    assert cache_sizes == sorted(cache_sizes)
    assert stream.get_kv_cache_size() >= cache_sizes[-1]
    _assert_stage_durations(stream, result)
    return result.texts, deltas


@pytest.mark.automatic_speech_recognition
@pytest.mark.real_models
def test_voxtral_parallel_streams(voxtral_pipeline, voxtral_audio_inputs):
    frame_sizes = (1600, 4800)
    expected = [
        _transcribe(voxtral_pipeline.create_stream(), audio, frame_samples)
        for audio, frame_samples in zip(voxtral_audio_inputs, frame_sizes)
    ]
    assert expected[0][0] != expected[1][0], "Distinct transcripts are required to detect session cross-talk."

    for _ in range(2):
        streams = [voxtral_pipeline.create_stream() for _ in voxtral_audio_inputs]
        first_delta_barrier = Barrier(len(streams))
        with ThreadPoolExecutor(max_workers=len(streams)) as executor:
            futures = [
                executor.submit(_transcribe, stream, audio, frame_samples, first_delta_barrier)
                for stream, audio, frame_samples in zip(streams, voxtral_audio_inputs, frame_sizes)
            ]
            actual = [future.result() for future in futures]
        assert actual == expected


@pytest.mark.automatic_speech_recognition
@pytest.mark.real_models
@pytest.mark.parametrize("profiling_enabled", [False, True])
@pytest.mark.parametrize("cross_audio_window", [False, True], ids=["short", "audio_window_eviction"])
def test_voxtral_kv_cache_size(voxtral_pipeline, profiling_enabled, cross_audio_window):
    model_dir = Path(os.environ["VOXTRAL_OV_MODEL"])
    config = json.loads((model_dir / "config.json").read_text())
    tekken = json.loads((model_dir / "tekken.json").read_text())
    audio_config = config["audio_config"]
    text_config = config["text_config"]
    preprocessor = json.loads((model_dir / "preprocessor_config.json").read_text())
    pipeline = (
        ov_genai.ASRPipeline(model_dir, "CPU", PERF_COUNT=True, INFERENCE_NUM_THREADS=4)
        if profiling_enabled
        else voxtral_pipeline
    )
    stream = pipeline.create_stream()
    assert stream.get_kv_cache_size() == 0
    assert stream.get_kv_cache_window_sizes() == {
        "audio": (0, audio_config["sliding_window"]),
        "text": (0, text_config["sliding_window"]),
    }
    assert all(duration == 0 for duration in stream.get_stage_durations().values())

    if cross_audio_window:
        num_samples = (
            (audio_config["sliding_window"] // config["downsample_factor"] + 2)
            * config.get("audio_length_per_tok", 8)
            * preprocessor["hop_length"]
        )
    else:
        # A short frame is processed by flush(), without transcript callbacks.
        num_samples = 1600
    with ThreadPoolExecutor(max_workers=1) as executor:
        generation = executor.submit(stream.generate)
        for offset in range(0, num_samples, 1600):
            stream.push_audio([0.0] * min(1600, num_samples - offset))
        stream.end_audio()
        result = generation.result()
    _assert_stage_durations(stream, result)
    durations = stream.get_stage_durations()
    for component in ("audio", "text"):
        operation_durations = [
            duration for stage, duration in durations.items() if stage.startswith(f"{component}_op/")
        ]
        if profiling_enabled:
            assert operation_durations
            assert sum(operation_durations) > 0
            broadcasts = {
                stage: duration for stage, duration in durations.items() if stage.startswith(f"{component}_node/")
            }
            assert broadcasts
            assert all(
                " compiled graph inputs=" in stage and " outputs=" in stage and " consumers=" in stage
                for stage in broadcasts
            )
            broadcast_groups = [
                duration for stage, duration in durations.items() if stage.startswith(f"{component}_op/Broadcast/")
            ]
            assert sum(broadcasts.values()) == pytest.approx(sum(broadcast_groups))
            assert stream.get_stage_durations() == durations
        else:
            assert not operation_durations
            assert not any(stage.startswith(f"{component}_node/") for stage in durations)
    generated_tokens = result.perf_metrics.get_num_generated_tokens()
    assert generated_tokens > 0

    text_positions = (
        tekken["audio"]["streaming_n_left_pad_tokens"] + config["default_num_delay_tokens"] + generated_tokens
    )
    audio_positions = text_positions * config["downsample_factor"]
    assert stream.get_kv_cache_window_sizes() == {
        "audio": (min(audio_positions, audio_config["sliding_window"]), audio_config["sliding_window"]),
        "text": (min(text_positions, text_config["sliding_window"]), text_config["sliding_window"]),
    }
    if cross_audio_window:
        assert audio_positions > audio_config["sliding_window"]
    else:
        assert audio_positions < audio_config["sliding_window"]
    audio_bytes = (
        2
        * audio_config["num_hidden_layers"]
        * audio_config["num_attention_heads"]
        * min(audio_positions, audio_config["sliding_window"])
        * audio_config["head_dim"]
        * 4
    )
    text_bytes = (
        2
        * text_config["num_hidden_layers"]
        * text_config["num_key_value_heads"]
        * min(text_positions, text_config["sliding_window"])
        * text_config["head_dim"]
        * 4
    )
    assert stream.get_kv_cache_size() == audio_bytes + text_bytes
