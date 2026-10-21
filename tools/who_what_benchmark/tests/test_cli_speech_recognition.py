# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform
import sys

import pytest

from conftest import convert_model, run_wwb
from test_cli_image import get_similarity

VLM_MODEL = "optimum-intel-internal-testing/tiny-random-gemma4"
FUNASR_MODEL = "optimum-intel-internal-testing/tiny-random-fun-asr"
FUNASR_LANGUAGE = "en"
QWEN3_ASR_MODEL = "optimum-intel-internal-testing/tiny-random-qwen3-asr"


def _common_args(gt_file):
    return ["--num-samples", "1", "--gt-data", gt_file, "--device", "CPU", "--model-type", "speech-recognition"]


@pytest.fixture(scope="module")
def asr_ground_truth(tmp_path_factory):
    gt_file = tmp_path_factory.mktemp("asr_ground_truth") / "gt.csv"
    run_wwb(["--base-model", VLM_MODEL, *_common_args(gt_file), "--hf"])
    return gt_file


@pytest.mark.skipif(
    sys.platform == "darwin" and platform.machine() == "arm64",
    reason="OpenVINO oneDNN ARM CPU backend can't compile the Gemma-4 audio encoder matmul. Ticket CVS-193092",
)
def test_asr_gemma4_optimum(tmp_path, asr_ground_truth):
    common = _common_args(asr_ground_truth)

    model_path = convert_model(VLM_MODEL)
    optimum_similarity = get_similarity(run_wwb(["--target-model", model_path, *common, "--output", tmp_path]))

    assert optimum_similarity >= 0.90


@pytest.fixture(scope="module")
def funasr_ground_truth(tmp_path_factory):
    gt_file = tmp_path_factory.mktemp("funasr_ground_truth") / "gt.csv"
    run_wwb(
        [
            "--base-model",
            FUNASR_MODEL,
            *_common_args(gt_file),
            "--hf",
            "--speech-language",
            FUNASR_LANGUAGE,
            "--max_new_tokens",
            "16",
        ]
    )
    return gt_file


@pytest.fixture(scope="module")
def qwen3_asr_ground_truth(tmp_path_factory):
    gt_file = tmp_path_factory.mktemp("qwen3_asr_ground_truth") / "gt.csv"
    run_wwb(
        [
            "--base-model",
            QWEN3_ASR_MODEL,
            *_common_args(gt_file),
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
def test_asr_optimum(tmp_path, request, model_id, ground_truth_fixture, language):
    gt_file = request.getfixturevalue(ground_truth_fixture)
    common = [*_common_args(gt_file), "--max_new_tokens", "16"]
    if language:
        common += ["--speech-language", language]

    model_path = convert_model(model_id)
    optimum_similarity = get_similarity(run_wwb(["--target-model", model_path, *common, "--output", tmp_path]))
    reproduced = get_similarity(run_wwb(["--target-data", tmp_path / "target.csv", *common]))

    assert reproduced == optimum_similarity
    if model_id == QWEN3_ASR_MODEL:
        assert optimum_similarity >= 0.90


def test_asr_funasr_genai(tmp_path, funasr_ground_truth):
    common = [*_common_args(funasr_ground_truth), "--speech-language", FUNASR_LANGUAGE, "--max_new_tokens", "16"]

    model_path = convert_model(FUNASR_MODEL)
    genai_similarity = get_similarity(run_wwb(["--target-model", model_path, *common, "--genai", "--output", tmp_path]))
    reproduced = get_similarity(run_wwb(["--target-data", tmp_path / "target.csv", *common]))

    assert reproduced == genai_similarity
