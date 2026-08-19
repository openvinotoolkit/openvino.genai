# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import types

import pytest

from wrappers.speech_to_text import FunASROptimumPipeline


class _FakeTokenizer:
    def __init__(self):
        self.batch_decode_calls = []

    def batch_decode(self, sequences, skip_special_tokens=False):
        self.batch_decode_calls.append((sequences, skip_special_tokens))
        return ["  hello world  "]


class _FakeFunASRModel:
    def __init__(self):
        self.preprocess_calls = []
        self.generate_calls = []

    def preprocess_input(self, sample, sampling_rate, **kwargs):
        import torch

        self.preprocess_calls.append({"sampling_rate": sampling_rate, **kwargs})
        return {
            "input_features": torch.zeros((1, 4)),
            "decoder_input_ids": torch.tensor([[10, 11, 12]]),
        }

    def generate(self, **kwargs):
        import torch

        self.generate_calls.append(kwargs)
        return torch.tensor([[10, 11, 12, 20, 21, 22]])


def test_funasr_wrapper_slices_generated_tokens_and_strips():
    model = _FakeFunASRModel()
    tokenizer = _FakeTokenizer()
    pipe = FunASROptimumPipeline(model=model, tokenizer=tokenizer)

    result = pipe(object(), generate_kwargs={"max_new_tokens": 16, "language": "en"})

    assert result["text"] == "hello world"
    assert result["language"] == "en"

    sequences, skip_special = tokenizer.batch_decode_calls[0]
    assert skip_special is True
    assert sequences.tolist() == [[20, 21, 22]]

    assert model.generate_calls[0]["max_new_tokens"] == 16
    assert model.preprocess_calls[0] == {"sampling_rate": 16000, "language": "en"}

    perf = result["perf_metrics"]
    assert set(perf) == {"preprocess_time", "generation_time", "detokenization_time"}


def test_funasr_wrapper_omits_language_when_not_provided():
    model = _FakeFunASRModel()
    pipe = FunASROptimumPipeline(model=model, tokenizer=_FakeTokenizer())

    result = pipe(object(), max_new_tokens=8)

    assert result["language"] is None
    assert model.preprocess_calls[0] == {"sampling_rate": 16000}


def test_funasr_wrapper_handles_sequences_output_attribute():
    model = _FakeFunASRModel()

    def generate_with_sequences(**kwargs):
        import torch

        return types.SimpleNamespace(sequences=torch.tensor([[10, 11, 12, 30, 31]]))

    model.generate = generate_with_sequences
    tokenizer = _FakeTokenizer()
    pipe = FunASROptimumPipeline(model=model, tokenizer=tokenizer)

    pipe(object(), generate_kwargs={"language": "en"})
    sequences, _ = tokenizer.batch_decode_calls[0]
    assert sequences.tolist() == [[30, 31]]


def test_model_detection_maps_fun_asr_to_speech_to_text():
    from llm_bench_utils.get_use_case import get_use_case_by_model_id

    use_case, model_type = get_use_case_by_model_id("fun-asr")
    assert use_case.task == "speech_to_text"
    assert model_type == "fun-asr"


def test_asr_hook_skips_incompatible_pipeline():
    from llm_bench_utils.hook_forward_whisper import ASRHook

    assert ASRHook().attach(types.SimpleNamespace(model=object())) is False


@pytest.mark.parametrize(
    ("model_type", "speech_param", "cli_language", "expected"),
    [
        ("fun-asr", {"language": "de"}, "en", "de"),
        ("fun-asr", {}, "de", "de"),
        ("fun-asr", {}, "  ", "en"),
        ("fun-asr", {}, "", "en"),
        ("qwen3-asr", {}, "", "English"),
        ("whisper", {}, "", "<|en|>"),
        ("whisper", {"language": "<|de|>"}, "en", "<|de|>"),
    ],
)
def test_resolve_speech_language_precedence(model_type, speech_param, cli_language, expected):
    from task.speech_to_text_generation import resolve_speech_language

    assert resolve_speech_language(model_type, speech_param, cli_language) == expected


@pytest.mark.parametrize(
    ("model_type", "speech_param", "expected"),
    [
        ("fun-asr", {}, False),
        ("fun-asr", {"timestamp": True}, True),
        ("whisper", {}, True),
        ("qwen3-asr", {}, True),
        ("whisper", {"timestamp": False}, False),
    ],
)
def test_resolve_return_timestamps(model_type, speech_param, expected):
    from task.speech_to_text_generation import resolve_return_timestamps

    assert resolve_return_timestamps(model_type, speech_param) is expected


def _has_asr_pipeline():
    import openvino_genai

    return hasattr(openvino_genai, "ASRPipeline")


@pytest.mark.skipif(
    _has_asr_pipeline(), reason="openvino_genai build provides ASRPipeline; missing-pipeline path is not exercised"
)
def test_genai_funasr_requires_asr_pipeline():
    from llm_bench_utils.config_class import UseCaseSpeech2Text
    from llm_bench_utils.ov_utils import create_genai_speech_2_txt_model

    use_case = UseCaseSpeech2Text(["fun-asr"])
    use_case.model_type = "fun-asr"

    with pytest.raises(RuntimeError, match="ASRPipeline"):
        create_genai_speech_2_txt_model(
            "unused-model-path",
            "CPU",
            memory_data_collector=None,
            processor=None,
            use_case=use_case,
            config={},
            mem_consumption=False,
        )
