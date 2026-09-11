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
