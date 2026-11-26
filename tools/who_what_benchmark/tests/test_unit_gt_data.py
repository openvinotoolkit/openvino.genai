# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from unittest.mock import Mock

import pandas as pd
from whowhatbench import wwb
from whowhatbench.registry import BaseEvaluator


class _DummyEvaluator(BaseEvaluator):
    """Holds fixed ground truth, so BaseEvaluator.dump_gt can write it without running a model."""

    def __init__(self):
        self.gt_data = pd.DataFrame({"prompts": ["Hello"], "answers": ["Hi"]})

    def score(self, model_or_data, **kwargs):
        raise NotImplementedError

    def worst_examples(self, top_k: int = 5, metric="similarity"):
        raise NotImplementedError

    def get_generation_fn(self):
        return None


def test_base_model_without_gt_data_saves_default_gt_data(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["wwb", "--base-model", "dummy-model"])
    monkeypatch.setattr(wwb, "load_model", Mock())
    monkeypatch.setattr(wwb, "create_evaluator", Mock(return_value=_DummyEvaluator()))

    with caplog.at_level(logging.WARNING):
        wwb.main()

    assert "--gt-data was not provided" in caplog.text
    assert pd.read_csv(tmp_path / wwb.DEFAULT_GT_DATA)["answers"].tolist() == ["Hi"]
