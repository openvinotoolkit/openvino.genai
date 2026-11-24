# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import functools
import hashlib
import json
import pathlib

import numpy as np
import openvino_genai as ov_genai
import pytest
import requests

from utils.atomic_download import AtomicDownloadManager
from utils.constants import get_ov_cache_downloaded_models_dir
from utils.dataset_utils import load_dataset_via_snapshot
from utils.network import retry_request

pytestmark = pytest.mark.automatic_speech_recognition

# Generated offline with the official Silero VAD v6.3 (see data/silero_vad/generate_silero_vad_reference.py).
REFERENCE = json.loads(
    (pathlib.Path(__file__).parent / "data/silero_vad/silero_vad_v6_3_reference.json").read_text(encoding="utf-8")
)
CASES = {case["id"]: case for case in REFERENCE["cases"]}
SAMPLE_RATE = 16000


@functools.lru_cache()
def silero_model_dir() -> pathlib.Path:
    source = REFERENCE["source"]
    model_dir = get_ov_cache_downloaded_models_dir() / f"silero_vad_{source['commit']}"

    def fetch() -> bytes:
        response = requests.get(source["model_url"], timeout=60)
        response.raise_for_status()
        return response.content

    AtomicDownloadManager(model_dir).execute(
        lambda temp_dir: (temp_dir / source["model"]).write_bytes(retry_request(fetch))
    )
    assert hashlib.sha256((model_dir / source["model"]).read_bytes()).hexdigest() == source["model_sha256"]
    return model_dir


@functools.lru_cache()
def recording() -> np.ndarray:
    audio = REFERENCE["audio"]
    dataset = load_dataset_via_snapshot(audio["dataset"], audio["config"], split=audio["split"])
    gap = np.zeros(audio["gap_samples"], dtype=np.float32)
    samples = np.concatenate(
        [
            part
            for index in audio["utterances"]
            for part in (np.asarray(dataset[index]["audio"]["array"], dtype=np.float32), gap)
        ]
    )
    assert hashlib.sha256(samples.tobytes()).hexdigest() == audio["sha256"], "LibriSpeech dummy audio changed"
    return samples


def case_audio(case: dict) -> np.ndarray:
    return recording()[case["first_sample"] : case["first_sample"] + case["num_samples"]]


def silero_vad_config(fields: dict) -> ov_genai.SileroVADConfig:
    config = ov_genai.SileroVADConfig()
    for name, value in fields.items():
        setattr(config, name, value)
    return config


def expected_timestamps(case: dict) -> list[tuple[float, float]]:
    return [(start / SAMPLE_RATE, end / SAMPLE_RATE) for start, end in case["segments"]]


def as_tuples(timestamps) -> list[tuple[float, float]]:
    return [(timestamp.start_ts, timestamp.end_ts) for timestamp in timestamps]


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["id"])
def test_matches_upstream_v6_3(case):
    detector = ov_genai.VoiceActivityDetector(silero_model_dir(), "CPU")
    timestamps = detector.get_speech_timestamps(case_audio(case), silero_vad_config=silero_vad_config(case["config"]))
    assert as_tuples(timestamps) == expected_timestamps(case)


def test_call_time_properties():
    detector = ov_genai.VoiceActivityDetector(silero_model_dir(), "CPU")
    default_case = CASES["multi_block"]
    ends_in_speech = case_audio(CASES["short_starts_in_speech"])

    first = detector.get_speech_timestamps(case_audio(default_case))
    detector.get_speech_timestamps(
        ends_in_speech, silero_vad_config=silero_vad_config(CASES["multi_block_all_options"]["config"])
    )
    second = detector.get_speech_timestamps(case_audio(default_case))

    assert as_tuples(first) == expected_timestamps(default_case)
    assert as_tuples(second) == expected_timestamps(default_case)


@pytest.mark.parametrize(
    "properties,message",
    [
        ({"max_speech_duration_s": 0.5}, "Unsupported Silero VAD property 'max_speech_duration_s'"),
        ({"silero_vad_config": silero_vad_config({"threshold": 1.0})}, "threshold must be in"),
    ],
    ids=["unsupported_property", "invalid_threshold"],
)
def test_rejects_invalid_properties(properties, message):
    detector = ov_genai.VoiceActivityDetector(silero_model_dir(), "CPU")
    with pytest.raises(RuntimeError, match=message):
        detector.get_speech_timestamps(case_audio(CASES["short_starts_in_speech"]), **properties)
