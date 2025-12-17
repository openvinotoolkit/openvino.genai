# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess  # nosec B404
import pytest
import logging
import sys
from test_cli_image import run_wwb
from profile_utils import _log, _stage


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_test(model_id, model_type, tmp_path, pruning_ratio, relevance_weight):
    if sys.platform == "darwin":
        pytest.xfail("Ticket 173169")
    if sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    _log(f"run_test: model_id={model_id} model_type={model_type} pruning_ratio={pruning_ratio} relevance_weight={relevance_weight}")
    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = tmp_path / model_id.replace("/", "_")

    with _stage("convert_model"):
        result = subprocess.run(
            [
                "optimum-cli",
                "export",
                "openvino",
                "-m",
                model_id,
                MODEL_PATH,
                "--task",
                "image-text-to-text",
                "--trust-remote-code",
            ],
            capture_output=True,
            text=True,
        )
    assert result.returncode == 0
    _log(f"Model converted to: {MODEL_PATH}")

    # Collect reference with HF model
    with _stage("run_wwb_hf_reference"):
        run_wwb(
        [
            "--base-model",
            model_id,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--hf",
        ]
    )

    env = {}
    env["OPENVINO_LOG_LEVEL"] = "7"
    # test cdpruner
    with _stage("run_wwb_cdpruner"):
        output = run_wwb(
        [
            "--target-model",
            MODEL_PATH,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
            "--output",
            tmp_path,
            "--pruning_ratio",
            pruning_ratio,
            "--relevance_weight",
            relevance_weight,
        ],
        env,
    )

    pruning_ratio_int = int(pruning_ratio)
    if pruning_ratio_int == 0:
        pruner_info = "pruning_ratio is 0, pruning disabled!"
    elif 0 < pruning_ratio_int < 100:
        pruner_info = f"Pruning Ratio: {pruning_ratio_int}%"
    elif pruning_ratio_int == 100:
        pruner_info = "Original visual tokens and pruned visual tokens are the same!"

    assert pruner_info in output


@pytest.mark.parametrize(
    ("model_id", "model_type", "pruning_ratio", "relevance_weight"),
    [
        ("optimum-intel-internal-testing/tiny-random-qwen2vl", "visual-text", "0", "0.8"),
        ("optimum-intel-internal-testing/tiny-random-qwen2vl", "visual-text", "20", "0.8"),
        ("optimum-intel-internal-testing/tiny-random-qwen2vl", "visual-text", "100", "0.8"),
    ],
)
def test_pruner_basic(model_id, model_type, tmp_path, pruning_ratio, relevance_weight):
    _log(f"test_pruner_basic: model_id={model_id} model_type={model_type} pruning_ratio={pruning_ratio} relevance_weight={relevance_weight}")
    run_test(model_id, model_type, tmp_path, pruning_ratio, relevance_weight)
