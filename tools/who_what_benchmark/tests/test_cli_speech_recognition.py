# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re

import pytest

from conftest import convert_model, run_wwb

ASR_MODEL = "optimum-intel-internal-testing/tiny-random-gemma4"


def get_wer_score(output: str) -> float:
    m = re.search(r"INFO:whowhatbench\.wwb:.*\bWER\b", output)
    assert m, "Could not find WER header in output"
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


def test_asr_gemma4(tmp_path):
    gt_file = tmp_path / "gt.csv"
    common = ["--num-samples", "1", "--gt-data", gt_file, "--device", "CPU", "--model-type", "speech-recognition"]

    run_wwb(["--base-model", ASR_MODEL, *common, "--hf"])

    if not _ov_gemma_audio_supported():
        pytest.skip("optimum-intel build lacks Gemma-4 OpenVINO audio export")

    model_path = convert_model(ASR_MODEL)
    optimum_wer = get_wer_score(run_wwb(["--target-model", model_path, *common, "--output", tmp_path]))
    genai_wer = get_wer_score(run_wwb(["--target-model", model_path, *common, "--genai", "--output", tmp_path]))
    reproduced = get_wer_score(run_wwb(["--target-data", tmp_path / "target.csv", *common]))

    assert optimum_wer >= 0.0 and genai_wer >= 0.0
    assert reproduced == genai_wer
