# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re

from conftest import run_wwb

ASR_MODEL = "optimum-intel-internal-testing/tiny-random-gemma4"


def get_wer_score(output: str) -> float:
    m = re.search(r"INFO:whowhatbench\.wwb:.*\bWER\b", output)
    assert m, "Could not find WER header in output"
    next_line = output[m.end() :].lstrip("\r").lstrip("\n").split("\n")[0]
    matches = re.findall(r"[-+]?\d*\.\d+", next_line)
    assert matches, f"Could not find score in line: {next_line!r}"
    return float(matches[-1])


def test_asr_gemma4_hf(tmp_path):
    gt_file = tmp_path / "gt.csv"
    common = [
        "--num-samples",
        "1",
        "--gt-data",
        gt_file,
        "--device",
        "CPU",
        "--model-type",
        "speech-recognition",
    ]

    run_wwb(["--base-model", ASR_MODEL, *common, "--hf"])

    output = run_wwb(["--target-model", ASR_MODEL, *common, "--hf", "--output", tmp_path])
    score = get_wer_score(output)

    output = run_wwb(["--target-data", tmp_path / "target.csv", *common])
    assert get_wer_score(output) == score
