# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from whowhatbench import model_loaders, wwb


@pytest.mark.parametrize(
    ("dtype_name", "expected_dtype"),
    [
        ("float32", torch.float32),
        ("fp32", torch.float32),
        ("float16", torch.float16),
        ("fp16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("bf16", torch.bfloat16),
    ],
)
def test_cli_dtype_reaches_hf_loader(monkeypatch, dtype_name, expected_dtype):
    monkeypatch.setattr(sys, "argv", ["wwb", "--gt-data", "gt.csv", "--hf", "--torch-dtype", dtype_name])
    args = wwb.parse_args()
    wwb.check_args(args)
    model_class = Mock()
    monkeypatch.setattr(
        model_loaders.AutoConfig, "from_pretrained", lambda *_: SimpleNamespace(quantization_config=None)
    )
    monkeypatch.setattr(model_loaders, "AutoModelForCausalLM", model_class)
    monkeypatch.setattr(model_loaders.torch.cuda, "is_available", lambda: False)

    model_loaders.load_model("text", "dummy-model", use_hf=args.hf, torch_dtype=args.torch_dtype)
    assert model_class.from_pretrained.call_args.kwargs["torch_dtype"] is expected_dtype


def test_unify_floating_dtype_casts_mixed_model():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2).to(torch.bfloat16))
    model = model_loaders._unify_floating_dtype(model, torch.float32)
    assert {p.dtype for p in model.parameters()} == {torch.float32}


@pytest.mark.parametrize("dtype", [None, "auto"])
def test_unify_floating_dtype_keeps_model_without_explicit_dtype(dtype):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2).to(torch.bfloat16))
    model = model_loaders._unify_floating_dtype(model, dtype)
    assert {p.dtype for p in model.parameters()} == {torch.float32, torch.bfloat16}


def test_unify_floating_dtype_skips_float8_storage():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2).to(torch.float8_e4m3fn))
    model = model_loaders._unify_floating_dtype(model, torch.float32)
    assert torch.float8_e4m3fn in {p.dtype for p in model.parameters()}


@pytest.mark.parametrize(
    "model_id, model_type, name_or_path, expected",
    [
        ("mistralai/Ministral-3-3B-Instruct-2512", "mistral3", "", True),
        ("/models/Ministral-3-3B-Instruct-2512/", "mistral3", "", True),
        ("C:\\models\\ministral-3-3b-instruct-2512", "mistral3", "", True),
        ("/models/local_dir", "mistral3", "mistralai/Ministral-3-3B-Instruct-2512", True),
        ("mistralai/Ministral-3-3B-Reasoning-2512", "mistral3", "", False),
        ("mistralai/Ministral-3-8B-Instruct-2512", "mistral3", "", False),
        ("other/Ministral-3-3B-Instruct-2512", "llava", "", False),
    ],
)
def test_floating_dtype_unification_limited_to_ministral_3_3b_instruct(model_id, model_type, name_or_path, expected):
    config = SimpleNamespace(model_type=model_type, _name_or_path=name_or_path)
    assert model_loaders._requires_floating_dtype_unification(model_id, config) is expected
