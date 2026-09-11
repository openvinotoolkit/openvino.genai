# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from transformers import SpeechT5ForTextToSpeech

from whowhatbench import model_loaders, speech_recognition_evaluator, wwb


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


def test_cli_rejects_torch_dtype_without_hf(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["wwb", "--gt-data", "gt.csv", "--torch-dtype", "float32"])
    with pytest.raises(ValueError, match="--torch-dtype requires --hf"):
        wwb.check_args(wwb.parse_args())


def test_speecht5_bfloat16_alignment(monkeypatch):
    model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    model.device, model.dtype = model.weight.device, model.weight.dtype
    model.generate = Mock(return_value=torch.ones(2, dtype=torch.bfloat16))
    vocoder = torch.nn.Linear(2, 2)
    processor = Mock(return_value={"input_ids": torch.tensor([[2, 3, 4]])})
    load = Mock(return_value=model)
    monkeypatch.setattr(SpeechT5ForTextToSpeech, "from_pretrained", load)
    monkeypatch.setattr(model_loaders, "_resolve_remote_code_and_config", lambda *_: (False, SimpleNamespace()))
    monkeypatch.setattr(model_loaders, "_load_speecht5_processor", lambda *_: processor)
    monkeypatch.setattr(model_loaders, "_load_speecht5_hifigan_vocoder", lambda *_: vocoder)

    wrapper = model_loaders.load_model("speech-generation", "dummy-speecht5", use_hf=True, torch_dtype="bf16")
    result = wrapper.generate("hello", speaker_embedding=torch.ones((1, 512)))

    assert load.call_args.kwargs["torch_dtype"] == vocoder.weight.dtype == torch.bfloat16
    assert model.generate.call_args.kwargs["speaker_embeddings"].dtype == torch.bfloat16
    assert result.speeches[0].data.dtype == np.float32


def test_funasr_casts_only_decoder(monkeypatch):
    decoder = torch.nn.Linear(2, 2, dtype=torch.float16)
    encoder = torch.nn.Linear(2, 2)
    auto_model = Mock(return_value=SimpleNamespace(model=SimpleNamespace(llm=decoder, audio_encoder=encoder)))
    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=auto_model))
    monkeypatch.setattr(speech_recognition_evaluator, "_get_asr_model_type", lambda *_: "funasr")

    model_loaders.load_model("speech-recognition", "dummy-funasr", use_hf=True, torch_dtype="bf16")

    kwargs = auto_model.call_args.kwargs
    assert kwargs["llm_dtype"] == "bf16"
    assert kwargs["fp16"] is False and kwargs["bf16"] is False
    assert decoder.weight.dtype == torch.bfloat16
    assert encoder.weight.dtype == torch.float32


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_kokoro_rejects_unsupported_dtype(monkeypatch, dtype_name):
    monkeypatch.setitem(sys.modules, "kokoro", None)

    with pytest.raises(ValueError, match="Use --torch-dtype fp32"):
        model_loaders.load_model("speech-generation", "dummy-kokoro", use_hf=True, torch_dtype=dtype_name)
