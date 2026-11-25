# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
import shutil
import ast
import json
import subprocess
import pandas as pd
import logging
from pathlib import Path
from test_cli_image import get_similarity
from conftest import convert_model, run_wwb


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def remove_artifacts(artifacts_path: Path):
    shutil.rmtree(artifacts_path)


@pytest.mark.parametrize(
    ("model_id", "threshold"),
    [
        ("cross-encoder/ms-marco-TinyBERT-L2-v2", 0.99),
        ("Qwen/Qwen3-Reranker-0.6B", 0.99),
        ("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", 0.99),
    ],
)
@pytest.mark.xfail(sys.platform == "darwin", reason="Hangs. Ticket 175534", run=False)
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


RERANKING_MODEL_ID = "cross-encoder/ms-marco-TinyBERT-L2-v2"


def write_reranking_csv(path, rows):
    pd.DataFrame(
        {
            "query": [query for query, _ in rows],
            "passages": [json.dumps(passages) for _, passages in rows],
        }
    ).to_csv(path, index=False)


def test_reranking_custom_dataset(tmp_path):
    rows = [
        ("How do I bake sourdough bread?", ["Mix flour, water and starter.", "Bake at 230 C for 40 minutes."]),
        ("What is the boiling point of water?", ["Water boils at 100 C at sea level.", "Ice melts at 0 C."]),
    ]
    dataset_file = tmp_path / "reranking.csv"
    write_reranking_csv(dataset_file, rows)
    gt_file = tmp_path / "gt.csv"

    run_wwb(
        [
            "--base-model",
            RERANKING_MODEL_ID,
            "--num-samples",
            "2",
            "--gt-data",
            gt_file,
            "--device",
            "CPU",
            "--model-type",
            "text-reranking",
            "--hf",
            "--dataset",
            dataset_file,
        ]
    )

    gt_data = pd.read_csv(gt_file, keep_default_na=False)
    assert list(gt_data["query"]) == [query for query, _ in rows]
    assert [ast.literal_eval(passages) for passages in gt_data["passages"]] == [passages for _, passages in rows]


def test_reranking_custom_dataset_invalid_passages(tmp_path):
    dataset_file = tmp_path / "reranking.csv"
    pd.DataFrame({"query": ["How do I bake sourdough bread?"], "passages": ["not json"]}).to_csv(dataset_file, index=False)

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_wwb(
            [
                "--base-model",
                RERANKING_MODEL_ID,
                "--num-samples",
                "1",
                "--gt-data",
                tmp_path / "gt.csv",
                "--device",
                "CPU",
                "--model-type",
                "text-reranking",
                "--hf",
                "--dataset",
                dataset_file,
            ]
        )
    assert "'passages' must be a JSON list of strings" in error.value.output


@pytest.mark.parametrize("ms_marco_format", [False, True])
def test_reranking_hf_dataset(tmp_path, ms_marco_format):
    rows = [
        ("How do I bake sourdough bread?", ["Mix flour, water and starter.", "Bake at 230 C for 40 minutes."]),
        ("What is the boiling point of water?", ["Water boils at 100 C at sea level.", "Ice melts at 0 C."]),
    ]
    # A local folder with a parquet file is loaded by datasets.load_dataset like a Hugging Face dataset.
    dataset_dir = tmp_path / "hf_dataset"
    dataset_dir.mkdir()
    pd.DataFrame(
        {
            "question": [query for query, _ in rows],
            "passages": [{"passage_text": passages} if ms_marco_format else passages for _, passages in rows],
        }
    ).to_parquet(dataset_dir / "train.parquet")
    gt_file = tmp_path / "gt.csv"

    run_wwb(
        [
            "--base-model",
            RERANKING_MODEL_ID,
            "--num-samples",
            "2",
            "--gt-data",
            gt_file,
            "--device",
            "CPU",
            "--model-type",
            "text-reranking",
            "--hf",
            "--dataset",
            dataset_dir,
            "--split",
            "train",
            "--dataset-field",
            "question",
        ]
    )

    gt_data = pd.read_csv(gt_file, keep_default_na=False)
    assert list(gt_data["query"]) == [query for query, _ in rows]
    assert [ast.literal_eval(passages) for passages in gt_data["passages"]] == [passages for _, passages in rows]
