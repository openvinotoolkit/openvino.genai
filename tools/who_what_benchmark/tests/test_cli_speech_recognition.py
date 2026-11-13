# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform
import sys
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download

from conftest import convert_model, run_wwb
from ov_utils import retry_request
from test_cli_image import get_similarity

VLM_MODEL = "optimum-intel-internal-testing/tiny-random-gemma4"
FUNASR_MODEL = "optimum-intel-internal-testing/tiny-random-fun-asr"
FUNASR_LANGUAGE = "en"
QWEN3_ASR_MODEL = "optimum-intel-internal-testing/tiny-random-qwen3-asr"
WHISPER_MODEL = "openai/whisper-tiny"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_VALIDATION_FILE = "parquet-data/en_us/validation-00000-of-00001.parquet"


@pytest.fixture(scope="module")
def fleurs_dataset():
    # A commit hash revision lets hf_hub_download return the cached file without Hub requests.
    dataset_file = retry_request(
        lambda: hf_hub_download("google/fleurs", FLEURS_VALIDATION_FILE, repo_type="dataset", revision=FLEURS_REVISION)
    )
    return str(Path(dataset_file).parent)


def _common_args(gt_file, fleurs_dataset):
    return [
        "--num-samples",
        "1",
        "--gt-data",
        gt_file,
        "--device",
        "CPU",
        "--model-type",
        "speech-recognition",
        "--dataset",
        fleurs_dataset,
    ]


@pytest.fixture(scope="module")
def asr_ground_truth(tmp_path_factory, fleurs_dataset):
    gt_file = tmp_path_factory.mktemp("asr_ground_truth") / "gt.csv"
    run_wwb(["--base-model", VLM_MODEL, *_common_args(gt_file, fleurs_dataset), "--hf"])
    return gt_file


@pytest.mark.skipif(
    sys.platform == "darwin" and platform.machine() == "arm64",
    reason="OpenVINO oneDNN ARM CPU backend can't compile the Gemma-4 audio encoder matmul. Ticket CVS-193092",
)
def test_asr_gemma4_optimum(tmp_path, asr_ground_truth, fleurs_dataset):
    common = _common_args(asr_ground_truth, fleurs_dataset)

    model_path = convert_model(VLM_MODEL)
    optimum_similarity = get_similarity(run_wwb(["--target-model", model_path, *common, "--output", tmp_path]))

    assert optimum_similarity >= 0.90


@pytest.fixture(scope="module")
def funasr_ground_truth(tmp_path_factory, fleurs_dataset):
    gt_file = tmp_path_factory.mktemp("funasr_ground_truth") / "gt.csv"
    run_wwb(
        [
            "--base-model",
            FUNASR_MODEL,
            *_common_args(gt_file, fleurs_dataset),
            "--hf",
            "--speech-language",
            FUNASR_LANGUAGE,
            "--max_new_tokens",
            "16",
        ]
    )
    return gt_file


@pytest.fixture(scope="module")
def qwen3_asr_ground_truth(tmp_path_factory, fleurs_dataset):
    gt_file = tmp_path_factory.mktemp("qwen3_asr_ground_truth") / "gt.csv"
    run_wwb(
        [
            "--base-model",
            QWEN3_ASR_MODEL,
            *_common_args(gt_file, fleurs_dataset),
            "--hf",
            "--max_new_tokens",
            "16",
        ]
    )
    return gt_file


@pytest.mark.parametrize(
    "model_id, ground_truth_fixture, language",
    [
        pytest.param(FUNASR_MODEL, "funasr_ground_truth", FUNASR_LANGUAGE, id="funasr"),
        pytest.param(QWEN3_ASR_MODEL, "qwen3_asr_ground_truth", "", id="qwen3_asr"),
    ],
)
def test_asr_optimum(tmp_path, request, model_id, ground_truth_fixture, language, fleurs_dataset):
    gt_file = request.getfixturevalue(ground_truth_fixture)
    common = [*_common_args(gt_file, fleurs_dataset), "--max_new_tokens", "16"]
    if language:
        common += ["--speech-language", language]

    model_path = convert_model(model_id)
    optimum_similarity = get_similarity(run_wwb(["--target-model", model_path, *common, "--output", tmp_path]))
    reproduced = get_similarity(run_wwb(["--target-data", tmp_path / "target.csv", *common]))

    assert reproduced == optimum_similarity


def test_asr_funasr_genai(tmp_path, funasr_ground_truth, fleurs_dataset):
    common = [
        *_common_args(funasr_ground_truth, fleurs_dataset),
        "--speech-language",
        FUNASR_LANGUAGE,
        "--max_new_tokens",
        "16",
    ]

    model_path = convert_model(FUNASR_MODEL)
    genai_similarity = get_similarity(run_wwb(["--target-model", model_path, *common, "--genai", "--output", tmp_path]))
    reproduced = get_similarity(run_wwb(["--target-data", tmp_path / "target.csv", *common]))

    assert reproduced == genai_similarity


@pytest.fixture(scope="module")
def whisper_ground_truth(tmp_path_factory, fleurs_dataset):
    gt_file = tmp_path_factory.mktemp("whisper_ground_truth") / "gt.csv"
    run_wwb(["--base-model", WHISPER_MODEL, *_common_args(gt_file, fleurs_dataset), "--hf"])
    return gt_file


def test_asr_whisper_optimum(tmp_path, whisper_ground_truth, fleurs_dataset):
    common = _common_args(whisper_ground_truth, fleurs_dataset)

    model_path = convert_model(WHISPER_MODEL)
    optimum_similarity = get_similarity(run_wwb(["--target-model", model_path, *common, "--output", tmp_path]))
    reproduced = get_similarity(run_wwb(["--target-data", tmp_path / "target.csv", *common]))

    assert optimum_similarity >= 0.90
    assert reproduced == optimum_similarity


def test_asr_whisper_genai(tmp_path, whisper_ground_truth, fleurs_dataset):
    common = _common_args(whisper_ground_truth, fleurs_dataset)

    model_path = convert_model(WHISPER_MODEL)
    genai_similarity = get_similarity(run_wwb(["--target-model", model_path, *common, "--genai", "--output", tmp_path]))
    reproduced = get_similarity(run_wwb(["--target-data", tmp_path / "target.csv", *common]))

    assert genai_similarity >= 0.90
    assert reproduced == genai_similarity
