# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import MODELS, convert_model, run_wwb
from test_cli_image import get_similarity
from whowhatbench.reranking_evaluator import is_qwen3_vl


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QWEN3_VL_RERANKER_MODEL_ID = "Qwen/Qwen3-VL-Reranker-2B"
MODELS.setdefault(
    "Qwen3-VL-Reranker-2B",
    {
        "name": QWEN3_VL_RERANKER_MODEL_ID,
        "convert_args": ["--trust-remote-code", "--task", "image-text-to-text"],
    },
)


def remove_artifacts(artifacts_path: Path):
    shutil.rmtree(artifacts_path)


def test_is_qwen3_vl():
    assert is_qwen3_vl(SimpleNamespace(model_type="qwen3_vl"))
    assert not is_qwen3_vl(SimpleNamespace(model_type="qwen3"))


@pytest.mark.wwb_rerank
@pytest.mark.parametrize(
    ("model_id", "threshold"),
    [
        ("cross-encoder/ms-marco-TinyBERT-L2-v2", 0.99),
        ("Qwen/Qwen3-Reranker-0.6B", 0.99),
        ("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", 0.99),
    ],
)
@pytest.mark.xfail(sys.platform == "darwin", reason="Hangs. Ticket 175534", run=False)
@pytest.mark.xfail(sys.platform == "win32", reason="Ticket 178790", run=False)
def test_reranking_optimum(model_id, threshold, tmp_path):
    GT_FILE = Path(tmp_path) / "gt.csv"
    MODEL_PATH = convert_model(model_id)

    # Collect reference with HF model
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
            "text-reranking",
            "--hf",
        ]
    )

    assert GT_FILE.exists()
    assert Path(tmp_path, "reference").exists()

    outputs_path = tmp_path / "optimum"
    # test Optimum
    outputs_optimum = run_wwb(
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
            "text-reranking",
            "--output",
            outputs_path,
        ]
    )

    assert (outputs_path / "target").exists()
    assert (outputs_path / "target.csv").exists()
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert "Metrics for model" in outputs_optimum

    similarity = get_similarity(outputs_optimum)
    assert similarity >= threshold

    remove_artifacts(outputs_path)

    outputs_path = tmp_path / "genai"
    # test GenAI
    outputs_genai = run_wwb(
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
            "text-reranking",
            "--genai",
            "--output",
            outputs_path,
        ]
    )
    assert (outputs_path / "target").exists()
    assert (outputs_path / "target.csv").exists()
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert "Metrics for model" in outputs_genai

    similarity = get_similarity(outputs_genai)
    assert similarity >= threshold

    # test w/o models
    run_wwb(
        [
            "--target-data",
            outputs_path / "target.csv",
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            "text-reranking",
            "--genai",
        ]
    )

    remove_artifacts(outputs_path)


@pytest.mark.wwb_rerank
@pytest.mark.skipif(
    os.environ.get("RUN_WWB_LARGE_MODELS", "false").lower() != "true",
    reason="Large Qwen3-VL reranker test is disabled by default",
)
@pytest.mark.xfail(sys.platform == "darwin", reason="Hangs. Ticket 175534", run=False)
@pytest.mark.xfail(sys.platform == "win32", reason="Ticket 178790", run=False)
def test_reranking_qwen3vl_optimum(tmp_path):
    gt_file = tmp_path / "gt.csv"
    model_path = convert_model(QWEN3_VL_RERANKER_MODEL_ID)

    run_wwb(
        [
            "--base-model",
            QWEN3_VL_RERANKER_MODEL_ID,
            "--num-samples",
            "1",
            "--gt-data",
            gt_file,
            "--device",
            "CPU",
            "--model-type",
            "text-reranking",
            "--hf",
        ]
    )

    assert gt_file.exists()
    assert (tmp_path / "reference").exists()

    outputs_path = tmp_path / "optimum"
    outputs_optimum = run_wwb(
        [
            "--target-model",
            model_path,
            "--num-samples",
            "1",
            "--gt-data",
            gt_file,
            "--device",
            "CPU",
            "--model-type",
            "text-reranking",
            "--output",
            outputs_path,
        ]
    )

    assert (outputs_path / "target").exists()
    assert (outputs_path / "target.csv").exists()
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert "Metrics for model" in outputs_optimum

    similarity = get_similarity(outputs_optimum)
    assert similarity >= 0.97

    remove_artifacts(outputs_path)
