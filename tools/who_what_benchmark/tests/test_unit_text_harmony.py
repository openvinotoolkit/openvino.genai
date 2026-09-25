# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest
import torch

from whowhatbench import wwb
from whowhatbench.chat_utils import is_harmony_template, is_native_harmony_prompt, normalize_chat_history
from whowhatbench.text_evaluator import TextEvaluator


class FakeInputs(dict):
    def to(self, _device):
        return self


class HarmonyTokenizer:
    chat_template = "<|start|>assistant<|channel|>final<|message|>"

    def __init__(self):
        self.applied_history = None
        self.raw_prompt = None
        self.add_special_tokens = None

    def apply_chat_template(self, history, **_kwargs):
        self.applied_history = history
        return FakeInputs(input_ids=torch.tensor([[1, 2]]))

    def __call__(self, prompt, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        self.raw_prompt = prompt
        self.add_special_tokens = add_special_tokens
        return FakeInputs(input_ids=torch.tensor([[1, 2]]))

    def batch_decode(self, _tokens, skip_special_tokens):
        assert skip_special_tokens
        return ["answer"]


class PlainTokenizer(HarmonyTokenizer):
    chat_template = "{{ messages }}"


class GenAIHarmonyTokenizer:
    def get_original_chat_template(self):
        return "<|start|>assistant<|channel|>final<|message|>"


class FakeModel:
    device = "cpu"

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.generation_input = None
        self.generate_kwargs = None

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, input_ids):
        return input_ids

    def generate(self, generation_input=None, **kwargs):
        self.generation_input = generation_input
        self.generate_kwargs = kwargs
        if isinstance(generation_input, list):
            return SimpleNamespace(texts=["answer"])
        return torch.tensor([[1, 2, 3]]) if generation_input is None else "answer"


@pytest.mark.parametrize("tokenizer", [HarmonyTokenizer(), GenAIHarmonyTokenizer()])
def test_detects_harmony_template(tokenizer):
    assert is_harmony_template(tokenizer)


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("<|user|>Question<|end|><|assistant|>", "Question"),
        ("<|system|>Rules<|end|><|user|>Question<|end|><|assistant|>", "Rules\r\n\r\nQuestion"),
        (
            "<|im_start|>system<|im_sep|>Rules<|im_end|>"
            "<|im_start|>user<|im_sep|>Question<|im_end|>"
            "<|im_start|>assistant<|im_sep|>",
            "Rules\r\n\r\nQuestion",
        ),
    ],
)
def test_normalizes_legacy_harmony_prompt(prompt, expected):
    assert normalize_chat_history(prompt, HarmonyTokenizer()) == [{"role": "user", "content": expected}]


def test_preserves_non_harmony_prompt():
    prompt = "<|user|>Question<|end|><|assistant|>"
    assert normalize_chat_history(prompt, PlainTokenizer()) == [{"role": "user", "content": prompt}]


@pytest.mark.parametrize(
    ("prompt", "expected_history", "expected_raw_prompt", "expected_add_special_tokens"),
    [
        ("Question", [{"role": "user", "content": "Question"}], None, None),
        ("<|user|>Question<|end|><|assistant|>", [{"role": "user", "content": "Question"}], None, None),
        (
            "<|start|>user<|message|>Question<|end|><|start|>assistant",
            None,
            "<|start|>user<|message|>Question<|end|><|start|>assistant",
            False,
        ),
    ],
)
def test_hf_text_generation_handles_harmony_prompt(
    prompt, expected_history, expected_raw_prompt, expected_add_special_tokens
):
    tokenizer = HarmonyTokenizer()
    TextEvaluator(
        base_model=FakeModel(),
        tokenizer=tokenizer,
        test_data=[prompt],
        metrics=(),
        use_chat_template=True,
    )

    assert tokenizer.applied_history == expected_history
    assert tokenizer.raw_prompt == expected_raw_prompt
    assert tokenizer.add_special_tokens is expected_add_special_tokens


@pytest.mark.parametrize(
    ("prompt", "expected_input", "apply_chat_template"),
    [
        ("Question", [{"role": "user", "content": "Question"}], True),
        ("<|user|>Question<|end|><|assistant|>", [{"role": "user", "content": "Question"}], True),
        (
            "<|start|>user<|message|>Question<|end|><|start|>assistant",
            "<|start|>user<|message|>Question<|end|><|start|>assistant",
            False,
        ),
    ],
)
def test_genai_text_generation_handles_harmony_prompt(monkeypatch, prompt, expected_input, apply_chat_template):
    monkeypatch.setitem(sys.modules, "openvino_genai", SimpleNamespace(ChatHistory=list))
    model = FakeModel(GenAIHarmonyTokenizer())

    assert wwb.genai_gen_text(model, None, prompt, 4, True, use_chat_template=True) == "answer"
    assert model.generation_input == expected_input
    assert model.generate_kwargs["apply_chat_template"] is apply_chat_template


def test_identifies_native_harmony_prompt():
    assert is_native_harmony_prompt(
        "<|start|>user<|message|>Question<|end|><|start|>assistant",
        HarmonyTokenizer(),
    )
