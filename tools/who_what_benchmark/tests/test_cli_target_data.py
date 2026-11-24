# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import Mock

import pandas as pd
import pytest

from whowhatbench import wwb


def test_base_model_can_be_compared_to_target_data_without_gt_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target_data = tmp_path / "target.csv"
    target_data.write_text("questions,answers\nHello,Hi\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["wwb", "--base-model", "base-model", "--target-data", str(target_data)])
    monkeypatch.setattr(wwb, "load_model", Mock())
    evaluator = Mock()
    metrics = pd.DataFrame({"similarity": [1.0]})
    evaluator.score.return_value = (metrics, metrics)
    monkeypatch.setattr(wwb, "create_evaluator", Mock(return_value=evaluator))

    wwb.main()

    evaluator.score.assert_called_once()
    assert evaluator.score.call_args.args[0] == str(target_data)
    assert pd.read_csv(tmp_path / "wwb_output" / "metrics.csv")["similarity"].tolist() == [1.0]


def test_missing_target_data_requires_target_model(tmp_path, monkeypatch):
    target_data = tmp_path / "missing.csv"
    monkeypatch.setattr(sys, "argv", ["wwb", "--base-model", "base-model", "--target-data", str(target_data)])

    with pytest.raises(ValueError, match=r"Target data file .* is not an existing file.*--target-model"):
        wwb.main()
