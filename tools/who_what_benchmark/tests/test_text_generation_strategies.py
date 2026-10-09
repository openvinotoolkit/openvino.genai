# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import logging
from pathlib import Path

import pandas as pd
import pytest

WWB_ROOT = Path(__file__).resolve().parents[1]
if str(WWB_ROOT) not in sys.path:
    sys.path.insert(0, str(WWB_ROOT))

from whowhatbench.text_generation_strategies import (  # noqa: E402
    CompositeGenerationStrategy,
    GenAISelfSufficientGeneration,
    GenerationStrategy,
    LlamaCPPReferenceBaseGenerationStrategy,
    LlamaCPPSelfSufficientGeneration,
    ReferenceBaseGenerationStrategy,
    SelfSufficientGenerationStrategy,
)

from whowhatbench.text_metrics_collection import Metrics  # noqa: E402

from conftest import run_wwb  # noqa: E402
from test_cli_text import base_model_path, target_model_path  # noqa: E402


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SIMILARITY = Metrics.SIMILARITY.value
KL_DIVERGENCY = Metrics.KL_DIVERGENCY.value
TOKEN_SIMILARITY = Metrics.TOKEN_SIMILARITY.value


@pytest.mark.parametrize(
    ("metrics_list", "is_genai", "is_llamacpp", "expected_cls"),
    [
        ([SIMILARITY], False, False, SelfSufficientGenerationStrategy),
        ([SIMILARITY], True, False, GenAISelfSufficientGeneration),
        ([SIMILARITY], False, True, LlamaCPPSelfSufficientGeneration),
        ([KL_DIVERGENCY], False, False, ReferenceBaseGenerationStrategy),
        ([KL_DIVERGENCY], False, True, LlamaCPPReferenceBaseGenerationStrategy),
    ],
)
def test_factory_creates_single_strategy(metrics_list, is_genai, is_llamacpp, expected_cls):
    strategy = GenerationStrategy.create(metrics_list, is_genai, is_llamacpp)

    assert type(strategy) is expected_cls


@pytest.mark.parametrize(
    ("is_llamacpp", "expected_classes"),
    [
        (False, [SelfSufficientGenerationStrategy, ReferenceBaseGenerationStrategy]),
        (True, [LlamaCPPSelfSufficientGeneration, LlamaCPPReferenceBaseGenerationStrategy]),
    ],
)
def test_factory_creates_composite_for_similarity_and_kl(is_llamacpp, expected_classes):
    strategy = GenerationStrategy.create([SIMILARITY, KL_DIVERGENCY], False, is_llamacpp, kld_ctx=8, kld_chunk=3)

    assert isinstance(strategy, CompositeGenerationStrategy)
    assert [type(inner) for inner in strategy.strategies] == expected_classes
    assert strategy.produced_fields == frozenset({"answer_text", "prompt_input_ids", "logits"})

    reference_strategy = strategy.strategies[1]
    assert reference_strategy.kld_ctx == 8
    assert reference_strategy.kld_chunk == 3


def _run_text_wwb(tmp_path, metrics_list, base_model=base_model_path, target_model=target_model_path, extra_args=()):
    gt_data_path = tmp_path / "gt.csv"
    output_dir = tmp_path / "target"
    output = run_wwb(
        [
            "--base-model",
            base_model,
            "--target-model",
            target_model,
            "--gt-data",
            gt_data_path,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--model-type",
            "text",
            "--short-prompt",
            "--output",
            output_dir,
            "--metrics",
            *metrics_list,
            *extra_args,
        ]
    )
    assert "Metrics for model" in output
    gt_data = pd.read_csv(gt_data_path, keep_default_na=False)
    metrics = pd.read_csv(output_dir / "metrics.csv")
    return gt_data, metrics


def _assert_npy_artifacts_exist(data, column):
    assert column in data.columns
    assert all(Path(path).exists() for path in data[column].values)


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    # Reference artifacts are stored relative to the working directory of the wwb process
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_cli_genai_self_sufficient_strategy(chdir_tmp):
    gt_data, metrics = _run_text_wwb(chdir_tmp, [SIMILARITY], extra_args=["--genai"])

    assert len(gt_data["answers"].values) == 1
    assert "similarity" in metrics.columns
    assert metrics["similarity"].values[0] > 0.99
    assert "kl_divergency" not in metrics.columns


def test_cli_reference_strategy_kl_divergency(tmp_path):
    gt_data, metrics = _run_text_wwb(tmp_path, [KL_DIVERGENCY])

    gt_data_path = tmp_path / "gt.csv"
    output_dir = tmp_path / "target"

    run_wwb(
        [
            "--base-model",
            base_model_path,
            "--gt-data",
            gt_data_path,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--model-type",
            "text",
            "--short-prompt",
            "--output",
            output_dir,
            "--metrics",
            KL_DIVERGENCY,
            TOKEN_SIMILARITY,
            SIMILARITY,
        ]
    )

    run_wwb(
        [
            "--target-model",
            target_model_path,
            "--gt-data",
            gt_data_path,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--model-type",
            "text",
            "--short-prompt",
            "--output",
            output_dir,
            "--metrics",
            KL_DIVERGENCY,
            TOKEN_SIMILARITY,
            SIMILARITY,
        ]
    )
    metrics = pd.read_csv(output_dir / "metrics.csv")
    assert SIMILARITY in metrics.columns
    assert KL_DIVERGENCY in metrics.columns

    assert metrics[SIMILARITY].values[0] > 0.99
    assert metrics[KL_DIVERGENCY].values[0] < 3.0

    run_wwb(
        [
            "--target-model",
            target_model_path,
            "--gt-data",
            gt_data_path,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--model-type",
            "text",
            "--short-prompt",
            "--output",
            output_dir,
            "--metrics",
            KL_DIVERGENCY,
            TOKEN_SIMILARITY,
            SIMILARITY,
        ]
    )

    metrics = pd.read_csv(output_dir / "metrics.csv")
    assert SIMILARITY in metrics.columns
    assert KL_DIVERGENCY in metrics.columns
    assert metrics[SIMILARITY].values[0] > 0.99
    assert metrics[KL_DIVERGENCY].values[0] < 3.0


def test_cli_composite_strategy_similarity_and_kl_divergency(chdir_tmp):
    gt_data, metrics = _run_text_wwb(chdir_tmp, [SIMILARITY, KL_DIVERGENCY, TOKEN_SIMILARITY])

    assert all(answer for answer in gt_data["answers"].values)
    _assert_npy_artifacts_exist(gt_data, "logits_path")
    _assert_npy_artifacts_exist(gt_data, "prompt_input_ids_path")
    _assert_npy_artifacts_exist(gt_data, "generated_token_ids_path")
    assert "similarity" in metrics.columns
    assert "kl_divergency" in metrics.columns
    assert "kl_divergency_p99" in metrics.columns
    assert "kl_divergency_p95" in metrics.columns
    assert "kl_divergency_p10" in metrics.columns
    assert "kl_divergency_min" in metrics.columns
    assert "kl_divergency_max" in metrics.columns
