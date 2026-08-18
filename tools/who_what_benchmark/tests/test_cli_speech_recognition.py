# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import re
import subprocess  # nosec B404

import pytest

from conftest import convert_model, run_wwb

VLM_MODEL = "optimum-intel-internal-testing/tiny-random-gemma4"
FUNASR_MODEL = "optimum-intel-internal-testing/tiny-random-fun-asr"
FUNASR_LANGUAGE = "en"


def _common_args(gt_file):
    return ["--num-samples", "1", "--gt-data", gt_file, "--device", "CPU", "--model-type", "speech-recognition"]


def get_similarity_score(output: str) -> float:
    m = re.search(r"INFO:whowhatbench\.wwb:.*\bsimilarity\b", output)
    assert m, "Could not find similarity header in output"
    next_line = output[m.end() :].lstrip("\r").lstrip("\n").split("\n")[0]
    matches = re.findall(r"[-+]?\d*\.\d+", next_line)
    assert matches, f"Could not find score in line: {next_line!r}"
    return float(matches[-1])


def _ov_gemma_audio_supported() -> bool:
    try:
        from optimum.exporters.openvino.model_configs import Gemma4ConfigBehavior

        return "audio_embeddings" in {behavior.value for behavior in Gemma4ConfigBehavior}
    except Exception:
        return False


def _funasr_supported() -> bool:
    return (
        importlib.util.find_spec("funasr") is not None
        and importlib.util.find_spec("optimum.intel.openvino.modeling_funasr") is not None
    )


@pytest.fixture(scope="module")
def asr_ground_truth(tmp_path_factory):
    gt_file = tmp_path_factory.mktemp("asr_ground_truth") / "gt.csv"
    run_wwb(["--base-model", VLM_MODEL, *_common_args(gt_file), "--hf"])
    return gt_file


def test_asr_gemma4_hf(tmp_path, asr_ground_truth):
    common = _common_args(asr_ground_truth)
    hf_similarity = get_similarity_score(run_wwb(["--target-model", VLM_MODEL, *common, "--hf", "--output", tmp_path]))
    reproduced = get_similarity_score(run_wwb(["--target-data", tmp_path / "target.csv", *common]))

    assert 0.0 <= hf_similarity <= 1.0
    assert reproduced == hf_similarity


@pytest.mark.skipif(not _ov_gemma_audio_supported(), reason="optimum-intel build lacks Gemma-4 OpenVINO audio export")
def test_asr_gemma4_optimum_genai(tmp_path, asr_ground_truth):
    common = _common_args(asr_ground_truth)

    model_path = convert_model(VLM_MODEL)
    optimum_similarity = get_similarity_score(run_wwb(["--target-model", model_path, *common, "--output", tmp_path]))
    genai_similarity = get_similarity_score(
        run_wwb(["--target-model", model_path, *common, "--genai", "--output", tmp_path])
    )
    reproduced = get_similarity_score(run_wwb(["--target-data", tmp_path / "target.csv", *common]))

    assert 0.0 <= optimum_similarity <= 1.0 and 0.0 <= genai_similarity <= 1.0
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


@pytest.mark.skipif(not _funasr_supported(), reason="requires the funasr package and FunASR support in optimum-intel")
def test_asr_funasr_optimum(tmp_path, funasr_ground_truth):
    common = [*_common_args(funasr_ground_truth), "--speech-language", FUNASR_LANGUAGE, "--max_new_tokens", "16"]

    model_path = convert_model(FUNASR_MODEL)
    optimum_similarity = get_similarity_score(run_wwb(["--target-model", model_path, *common, "--output", tmp_path]))
    reproduced = get_similarity_score(run_wwb(["--target-data", tmp_path / "target.csv", *common]))

    assert 0.0 <= optimum_similarity <= 1.0
    assert reproduced == optimum_similarity


@pytest.mark.skipif(not _funasr_supported(), reason="requires the funasr package and FunASR support in optimum-intel")
def test_asr_funasr_genai(tmp_path, funasr_ground_truth):
    common = [*_common_args(funasr_ground_truth), "--speech-language", FUNASR_LANGUAGE, "--max_new_tokens", "16"]

    model_path = convert_model(FUNASR_MODEL)
    try:
        genai_similarity = get_similarity_score(
            run_wwb(["--target-model", model_path, *common, "--genai", "--output", tmp_path])
        )
    except subprocess.CalledProcessError as error:
        if "fun_asr" in (error.output or "") and "Unsupported" in (error.output or ""):
            pytest.skip("Installed OpenVINO GenAI build has no FunASR support in ASRPipeline")
        raise
    reproduced = get_similarity_score(run_wwb(["--target-data", tmp_path / "target.csv", *common]))

    assert 0.0 <= genai_similarity <= 1.0
    assert reproduced == genai_similarity
