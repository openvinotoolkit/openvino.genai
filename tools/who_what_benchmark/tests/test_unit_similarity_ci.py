# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from whowhatbench import wwb
from whowhatbench.whowhat_metrics import bootstrap_mean_ci, similarity_summary


def test_bootstrap_ci_contains_mean_and_is_reproducible():
    values = np.random.default_rng(0).uniform(0.5, 1.0, size=30)

    low, high = bootstrap_mean_ci(values)

    assert low < values.mean() < high
    assert bootstrap_mean_ci(values) == (low, high)


def test_bootstrap_ci_matches_normal_approximation():
    # For a large sample the bootstrap interval should be close to mean +- 1.96 * standard error.
    values = np.random.default_rng(1).normal(0.9, 0.05, size=2000)
    half_width = 1.96 * values.std(ddof=1) / np.sqrt(values.size)

    low, high = bootstrap_mean_ci(values)

    assert low == pytest.approx(values.mean() - half_width, abs=5e-4)
    assert high == pytest.approx(values.mean() + half_width, abs=5e-4)


def test_bootstrap_ci_is_zero_width_for_identical_values():
    assert bootstrap_mean_ci([1.0, 1.0, 1.0]) == (1.0, 1.0)


def test_bootstrap_ci_needs_two_values():
    assert bootstrap_mean_ci([]) is None
    assert bootstrap_mean_ci([0.9]) is None


def test_similarity_summary_leaves_out_non_finite_values():
    summary = similarity_summary([0.9, float("nan"), 0.7, 1.0])

    assert summary["similarity_min"] == pytest.approx(0.7)
    assert summary["num_samples"] == 3
    assert summary["num_non_finite"] == 1
    assert summary["similarity_ci_low"] <= summary["similarity_ci_high"]


def test_cli_logs_ci_before_table_and_saves_it(tmp_path, monkeypatch, caplog):
    per_question = pd.DataFrame({"similarity": [0.6, 0.8, 0.9, 1.0]})
    metrics = pd.DataFrame([{"similarity": per_question["similarity"].mean()}])
    evaluator = Mock()
    evaluator.score.return_value = (per_question, metrics)
    target_data = tmp_path / "target.csv"
    target_data.write_text("prompts,answers\nHello,Hi\n", encoding="utf-8")
    gt_data = tmp_path / "gt.csv"
    gt_data.write_text("prompts,answers\nHello,Hi\n", encoding="utf-8")
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["wwb", "--gt-data", str(gt_data), "--target-data", str(target_data), "--output", str(output)],
    )
    monkeypatch.setattr(wwb, "create_evaluator", Mock(return_value=evaluator))

    with caplog.at_level(logging.INFO):
        wwb.main()

    messages = [record.getMessage() for record in caplog.records]
    ci_index = next(i for i, m in enumerate(messages) if "confidence interval" in m)
    table_index = next(i for i, m in enumerate(messages) if m.lstrip().startswith("similarity"))
    assert ci_index < table_index

    saved = pd.read_csv(output / "metrics.csv", index_col=0)
    assert list(saved.columns) == ["similarity", "similarity_ci_low", "similarity_ci_high", "similarity_min"]
    assert saved["similarity_min"].iloc[0] == pytest.approx(0.6)
    assert saved["similarity_ci_low"].iloc[0] <= saved["similarity"].iloc[0] <= saved["similarity_ci_high"].iloc[0]
