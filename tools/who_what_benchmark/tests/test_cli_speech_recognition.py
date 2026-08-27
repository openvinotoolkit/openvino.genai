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


def _common_args(gt_file):
    return ["--num-samples", "1", "--gt-data", gt_file, "--device", "CPU", "--model-type", "speech-recognition"]


@pytest.fixture(scope="module")
def asr_ground_truth(tmp_path_factory):
    gt_file = tmp_path_factory.mktemp("asr_ground_truth") / "gt.csv"
    run_wwb(["--base-model", VLM_MODEL, *_common_args(gt_file), "--hf"])
    return gt_file


def test_asr_gemma4_hf(tmp_path, asr_ground_truth):
    common = _common_args(asr_ground_truth)
    hf_similarity = get_similarity(run_wwb(["--target-model", VLM_MODEL, *common, "--hf", "--output", tmp_path]))
    reproduced = get_similarity(run_wwb(["--target-data", tmp_path / "target.csv", *common]))

    assert hf_similarity == 1.0
    assert reproduced == hf_similarity


@pytest.mark.skipif(
    sys.platform == "darwin" and platform.machine() == "arm64",
    reason="OpenVINO oneDNN ARM CPU backend can't compile the Gemma-4 audio encoder matmul. Ticket CVS-193092",
)
def test_asr_gemma4_optimum_genai(tmp_path, asr_ground_truth):
    common = _common_args(asr_ground_truth)

    model_path = convert_model(VLM_MODEL)
    optimum_similarity = get_similarity(run_wwb(["--target-model", model_path, *common, "--output", tmp_path]))
    # GenAI audio backend is not supported yet
    # genai_similarity = get_similarity(run_wwb(["--target-model", model_path, *common, "--genai", "--output", tmp_path]))
    reproduced = get_similarity(run_wwb(["--target-data", tmp_path / "target.csv", *common]))

    assert optimum_similarity >= 0.90 and genai_similarity >= 0.90
    assert reproduced == genai_similarity


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


def test_asr_funasr_optimum(tmp_path, funasr_ground_truth):
    common = [*_common_args(funasr_ground_truth), "--speech-language", FUNASR_LANGUAGE, "--max_new_tokens", "16"]

    model_path = convert_model(FUNASR_MODEL)
    optimum_similarity = get_similarity(run_wwb(["--target-model", model_path, *common, "--output", tmp_path]))
    reproduced = get_similarity(run_wwb(["--target-data", tmp_path / "target.csv", *common]))

    assert optimum_similarity >= 0.90
    assert reproduced == optimum_similarity


def test_asr_funasr_genai(tmp_path, funasr_ground_truth):
    common = [*_common_args(funasr_ground_truth), "--speech-language", FUNASR_LANGUAGE, "--max_new_tokens", "16"]

    model_path = convert_model(FUNASR_MODEL)
    genai_similarity = get_similarity(run_wwb(["--target-model", model_path, *common, "--genai", "--output", tmp_path]))
    reproduced = get_similarity(run_wwb(["--target-data", tmp_path / "target.csv", *common]))

    assert genai_similarity >= 0.90
    assert reproduced == genai_similarity
