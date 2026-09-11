# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from transformers import SpeechT5Config, SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5HifiGanConfig

from whowhatbench import model_loaders, speech_generation_evaluator, speech_recognition_evaluator, wwb


@pytest.mark.parametrize("dtype_name", [None, "bfloat16"])
def test_diffusers_loader_dtype(monkeypatch, dtype_name):
    from diffusers import DiffusionPipeline

    load = Mock()
    monkeypatch.setattr(DiffusionPipeline, "from_pretrained", load)
    model_loaders.load_model("text-to-image", "dummy-model", use_hf=True, torch_dtype=dtype_name)
    assert load.call_args.kwargs["torch_dtype"] == (getattr(torch, dtype_name) if dtype_name else torch.float32)


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


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_hf_speecht5_tiny_generation_dtype(monkeypatch, dtype_name):
    expected_dtype = getattr(torch, dtype_name)
    config = SpeechT5Config(
        hidden_size=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        speech_decoder_prenet_units=16,
        speech_decoder_postnet_units=16,
    )
    vocoder = SpeechT5HifiGan(
        SpeechT5HifiGanConfig(
            upsample_initial_channel=16,
            upsample_rates=[2],
            upsample_kernel_sizes=[4],
            resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 3, 5]],
        )
    ).eval()

    def from_pretrained(model_id, **kwargs):
        model = SpeechT5ForTextToSpeech(config).eval().to(dtype=kwargs["torch_dtype"])
        model.generate = partial(model.generate, maxlenratio=1.0)
        return model

    def processor(**kwargs):
        return {"input_ids": torch.tensor([[2, 3, 4]])}

    monkeypatch.setattr(SpeechT5ForTextToSpeech, "from_pretrained", from_pretrained)
    monkeypatch.setattr(model_loaders, "_resolve_remote_code_and_config", lambda *_: (False, config))
    monkeypatch.setattr(model_loaders, "_load_speecht5_processor", lambda *_: processor)
    monkeypatch.setattr(model_loaders, "_load_speecht5_hifigan_vocoder", lambda *_: vocoder)

    wrapper = model_loaders.load_model("speech-generation", "tiny-speecht5", use_hf=True, torch_dtype=dtype_name)
    result = wrapper.generate("hello", speaker_embedding=torch.ones((1, 512)))
    audio = result.speeches[0].data

    assert wrapper.model.dtype == expected_dtype
    assert vocoder.dtype == expected_dtype
    assert audio.size > 0
    assert np.isfinite(audio).all()
    assert audio.dtype == np.float32


def test_speecht5_optimum_inputs_remain_native():
    generate = Mock(return_value=torch.zeros(2))
    wrapper = speech_generation_evaluator.SpeechT5Wrapper(
        SimpleNamespace(device="CPU", dtype="f32", generate=generate),
        lambda **kwargs: {"input_ids": torch.tensor([[2, 3, 4]])},
        None,
    )
    wrapper.generate("hello", speaker_embedding=torch.ones((1, 512)))
    assert generate.call_args.args[0].dtype == torch.int64
    assert generate.call_args.kwargs["speaker_embeddings"].dtype == torch.float32


def test_funasr_float32_weights(monkeypatch):
    model = torch.nn.Linear(2, 2, dtype=torch.float16)
    auto_model = Mock(return_value=SimpleNamespace(model=model))
    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=auto_model))
    monkeypatch.setattr(speech_recognition_evaluator, "_get_asr_model_type", lambda *_: "funasr")

    model_loaders.load_model("speech-recognition", "dummy-funasr", use_hf=True, torch_dtype="float32")

    assert model.weight.dtype == torch.float32
    precision = {
        key: value for key, value in auto_model.call_args.kwargs.items() if key in ("fp16", "bf16", "llm_dtype")
    }
    assert precision == {"fp16": False, "bf16": False, "llm_dtype": "fp32"}


def test_kokoro_float32_weights(monkeypatch):
    model = torch.nn.Linear(2, 2, dtype=torch.float16)
    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=Mock()))
    monkeypatch.setitem(sys.modules, "kokoro.model", SimpleNamespace(KModel=lambda **kwargs: model))

    model_loaders.load_model("speech-generation", "dummy-kokoro", use_hf=True, torch_dtype="float32")

    assert model.weight.dtype == torch.float32


@pytest.mark.parametrize(
    ("model_type", "model_id", "module_name"),
    [
        ("speech-recognition", "dummy-funasr", "funasr"),
        ("speech-generation", "dummy-kokoro", "kokoro"),
    ],
)
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_source_speech_rejects_unsupported_dtype_before_import(
    monkeypatch, model_type, model_id, module_name, dtype_name
):
    monkeypatch.setitem(sys.modules, module_name, None)
    monkeypatch.setattr(speech_recognition_evaluator, "_get_asr_model_type", lambda *_: "funasr")

    with pytest.raises(ValueError, match="support only float32 for --torch-dtype"):
        model_loaders.load_model(model_type, model_id, use_hf=True, torch_dtype=dtype_name)
