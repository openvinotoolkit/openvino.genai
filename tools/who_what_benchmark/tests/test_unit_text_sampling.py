# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest
import torch

from whowhatbench import wwb
from whowhatbench.chat_text_evaluator import default_gen_answer as default_chat_gen_answer
from whowhatbench.text_evaluator import TextEvaluator


class FakeInputs(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    def __call__(self, _prompt, return_tensors):
        assert return_tensors == "pt"
        return FakeInputs(input_ids=torch.tensor([[1, 2]]))

    def apply_chat_template(self, _history, **_kwargs):
        return FakeInputs(input_ids=torch.tensor([[1, 2]]))

    def batch_decode(self, _tokens, skip_special_tokens):
        assert skip_special_tokens
        return ["answer"]


class FakeModel:
    device = "cpu"

    def __init__(self, chat=False):
        self.chat = chat
        self.generate_kwargs = None

    def forward(self, input_ids):
        return input_ids

    def generate(self, *_args, **kwargs):
        self.generate_kwargs = kwargs
        sequences = torch.tensor([[1, 2, 3]])
        if self.chat:
            return SimpleNamespace(sequences=sequences, past_key_values=None)
        return sequences


class FakeGenAIChatModel:
    def __init__(self):
        self.generate_kwargs = None

    def generate(self, *_args, **kwargs):
        self.generate_kwargs = kwargs
        return SimpleNamespace(texts=["answer"])


@pytest.mark.parametrize(
    ("flag", "expected"),
    [([], None), (["--do-sample"], True), (["--do_sample"], True), (["--greedy"], False)],
)
def test_sampling_cli(monkeypatch, flag, expected):
    monkeypatch.setattr(sys, "argv", ["wwb", *flag])
    assert wwb.parse_args().do_sample is expected


def test_sampling_cli_rejects_conflicting_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["wwb", "--do-sample", "--greedy"])
    with pytest.raises(SystemExit):
        wwb.parse_args()


@pytest.mark.parametrize("do_sample", [None, True, False])
def test_text_generation_sampling_override(do_sample):
    model = FakeModel()
    generation_config_extra = {} if do_sample is None else {"do_sample": do_sample}

    TextEvaluator(
        base_model=model,
        tokenizer=FakeTokenizer(),
        test_data=["prompt"],
        metrics=(),
        generation_config_extra=generation_config_extra,
    )

    if do_sample is None:
        assert "do_sample" not in model.generate_kwargs
    else:
        assert model.generate_kwargs["do_sample"] is do_sample


@pytest.mark.parametrize("do_sample", [None, True, False])
def test_chat_generation_sampling_override(do_sample):
    model = FakeModel(chat=True)
    generation_config_extra = {} if do_sample is None else {"do_sample": do_sample}

    default_chat_gen_answer(
        model,
        FakeTokenizer(),
        ["prompt"],
        max_new_tokens=4,
        generation_config_extra=generation_config_extra,
    )

    if do_sample is None:
        assert "do_sample" not in model.generate_kwargs
    else:
        assert model.generate_kwargs["do_sample"] is do_sample


@pytest.mark.parametrize("do_sample", [None, True, False])
def test_genai_generation_sampling_override(do_sample):
    model = FakeModel()
    generation_config_extra = {} if do_sample is None else {"do_sample": do_sample}

    wwb.genai_gen_text(
        model,
        tokenizer=None,
        question="prompt",
        max_new_tokens=4,
        skip_question=True,
        generation_config_extra=generation_config_extra,
    )

    if do_sample is None:
        assert "do_sample" not in model.generate_kwargs
    else:
        assert model.generate_kwargs["do_sample"] is do_sample


@pytest.mark.parametrize("do_sample", [None, True, False])
def test_genai_chat_generation_sampling_override(monkeypatch, do_sample):
    monkeypatch.setitem(sys.modules, "openvino_genai", SimpleNamespace(ChatHistory=list))
    model = FakeGenAIChatModel()
    generation_config_extra = {} if do_sample is None else {"do_sample": do_sample}

    wwb.genai_gen_chat_text(
        model,
        _tokenizer=None,
        prompts=["prompt"],
        max_new_tokens=4,
        generation_config_extra=generation_config_extra,
    )

    if do_sample is None:
        assert "do_sample" not in model.generate_kwargs
    else:
        assert model.generate_kwargs["do_sample"] is do_sample
