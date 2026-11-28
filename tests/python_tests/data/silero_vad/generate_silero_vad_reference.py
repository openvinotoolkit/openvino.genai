# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Offline utility for manually regenerating the committed Silero VAD reference JSON with the official
# Silero VAD v6.3 implementation on ONNX Runtime (pip install silero-vad==6.3 onnxruntime datasets).
# CI never runs this script; test_voice_activity_detector.py compares GenAI against the committed JSON.

import hashlib
import json
import pathlib
from importlib import metadata, resources

import datasets
import numpy as np
from huggingface_hub import snapshot_download
from silero_vad import load_silero_vad
from silero_vad.sequence_vad import get_speech_timestamps_sequence

SILERO_VAD_VERSION = "6.3"
SILERO_VAD_COMMIT = "63d4a2a7bfce8cd9bb39ed8fa0afcb5236d73e01"
MODEL_FILE = "silero_vad_16k_sequence.onnx"
MODEL_SHA256 = "b8918bcf4e9e13870dd43a6eda22ab4faaa0afcb22b9c7cd6bc85900d30cffca"
MODEL_URL = (
    f"https://raw.githubusercontent.com/snakers4/silero-vad/{SILERO_VAD_COMMIT}/src/silero_vad/data/{MODEL_FILE}"
)
DATASET = "hf-internal-testing/librispeech_asr_dummy"
DATASET_CONFIG = "clean"
DATASET_SPLIT = "validation"
DATASET_REVISION = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"
UTTERANCES = [0, 1, 2, 3]
GAP_SAMPLES = 8000
UPSTREAM_OPTION_NAMES = {"min_silence_at_max_speech_ms": "min_silence_at_max_speech"}
ALL_OPTIONS = {
    "threshold": 0.6,
    "neg_threshold": 0.3,
    "min_speech_duration_ms": 100,
    "min_silence_duration_ms": 300,
    "speech_pad_ms": 100,
    "max_speech_duration_s": 5.0,
    "min_silence_at_max_speech_ms": 50,
}
CASES = [
    {"id": "short_starts_in_speech", "first_sample": 16000, "num_samples": 24100, "config": {}},
    {"id": "first_block_plus_one_sample", "first_sample": 0, "num_samples": 512 * 512 + 1, "config": {}},
    {"id": "multi_block", "first_sample": 0, "num_samples": None, "config": {}},
    {
        "id": "multi_block_max_speech_1s",
        "first_sample": 0,
        "num_samples": None,
        "config": {"max_speech_duration_s": 1.0},
    },
    {"id": "multi_block_all_options", "first_sample": 0, "num_samples": None, "config": ALL_OPTIONS},
    {
        "id": "longest_pause_split",
        "first_sample": 367616,
        "num_samples": None,
        "config": {"min_silence_duration_ms": 2000, "max_speech_duration_s": 7.5},
    },
]
# Probabilities closer than this to a threshold could flip between ONNX Runtime and OpenVINO.
MIN_THRESHOLD_MARGIN = 1e-3


def recording():
    dataset_path = snapshot_download(DATASET, repo_type="dataset", revision=DATASET_REVISION)
    dataset = datasets.load_dataset(dataset_path, DATASET_CONFIG, split=DATASET_SPLIT)
    gap = np.zeros(GAP_SAMPLES, dtype=np.float32)
    return np.concatenate(
        [part for index in UTTERANCES for part in (np.asarray(dataset[index]["audio"]["array"], dtype=np.float32), gap)]
    )


if __name__ == "__main__":
    assert metadata.version("silero-vad") == SILERO_VAD_VERSION
    model_bytes = (resources.files("silero_vad") / "data" / MODEL_FILE).read_bytes()
    assert hashlib.sha256(model_bytes).hexdigest() == MODEL_SHA256
    model = load_silero_vad(sequence=True)
    audio = recording()

    cases = []
    for case in CASES:
        samples = audio[case["first_sample"] :][: case["num_samples"]]
        config = case["config"]
        threshold = config.get("threshold", 0.5)
        neg_threshold = config.get("neg_threshold", max(threshold - 0.15, 0.01))
        probabilities = model.audio_forward(samples)
        margin = min(np.abs(probabilities - threshold).min(), np.abs(probabilities - neg_threshold).min())
        assert margin >= MIN_THRESHOLD_MARGIN, (case["id"], margin)
        upstream_config = {UPSTREAM_OPTION_NAMES.get(name, name): value for name, value in config.items()}
        segments = get_speech_timestamps_sequence(samples, model, **upstream_config)
        cases.append(
            {
                "id": case["id"],
                "first_sample": case["first_sample"],
                "num_samples": int(samples.size),
                "config": config,
                "segments": [[segment["start"], segment["end"]] for segment in segments],
            }
        )

    reference = {
        "source": {
            "silero_vad": SILERO_VAD_VERSION,
            "commit": SILERO_VAD_COMMIT,
            "model": MODEL_FILE,
            "model_url": MODEL_URL,
            "model_sha256": MODEL_SHA256,
            "reference": "silero_vad.sequence_vad.get_speech_timestamps_sequence",
        },
        "audio": {
            "dataset": DATASET,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "revision": DATASET_REVISION,
            "utterances": UTTERANCES,
            "gap_samples": GAP_SAMPLES,
            "num_samples": int(audio.size),
            "sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
        },
        "cases": cases,
    }
    out_path = pathlib.Path(__file__).parent / "silero_vad_v6_3_reference.json"
    out_path.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
