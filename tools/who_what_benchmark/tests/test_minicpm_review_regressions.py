# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from whowhatbench import wwb


@pytest.mark.parametrize("split", [None, "test"])
def test_load_prompts_default_split(monkeypatch, split):
    load_dataset = Mock(return_value={"text": ["question"]})
    monkeypatch.setattr(wwb, "load_dataset", load_dataset)
    args = Namespace(dataset="org/dataset", split=split, dataset_field="text")

    assert wwb.load_prompts(args) == {"prompts": ["question"]}
    load_dataset.assert_called_once_with(path="org/dataset", name=None, split=split or "validation")


@pytest.mark.parametrize(
    "model_type,automatic_crop",
    [("minicpmv", True), ("minicpmo", True), ("minicpmv4_6", False), ("minicpm_ocr", False)],
)
def test_minicpm_automatic_crop(model_type, automatic_crop):
    assert wwb.is_model_with_automatic_crop(SimpleNamespace(model_type=model_type)) is automatic_crop
