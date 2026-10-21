# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest
import torch
import numpy as np

from whowhatbench import wwb
from whowhatbench.artifacts_manager import Metrics
from whowhatbench.chat_utils import is_harmony_template, is_native_harmony_prompt, normalize_chat_history
from whowhatbench.text_evaluator import TextEvaluator
from whowhatbench.text_generation_strategies import GenAIGeneration, KLDivergencyGeneration, OptimumHFGeneration


HARMONY_TEMPLATE = (
    "{{- '<|start|>' + message['role'] -}}"
    "{% if message['channel'] is defined %}{{- '<|meta_sep|>' + message['channel'] -}}{% endif %}"
    "{{- '<|message|>' + message['content'] + '<|end|>' -}}"
)


class FakeInputs(dict):
    def to(self, _device):
        return self


class HarmonyTokenizer:
    chat_template = HARMONY_TEMPLATE

    def __init__(self):
        self.applied_history = None
        self.raw_prompt = None
        self.add_special_tokens = None

    def apply_chat_template(self, history, **_kwargs):
        self.applied_history = history
        return FakeInputs(input_ids=torch.tensor([[1, 2]]))

    def __call__(self, prompt, return_tensors, add_special_tokens=True):
        assert return_tensors == "pt"
        self.raw_prompt = prompt
        self.add_special_tokens = add_special_tokens
        return FakeInputs(input_ids=torch.tensor([[1, 2]]))

    def batch_decode(self, _tokens, skip_special_tokens):
        assert skip_special_tokens
        return ["answer"]


class PlainTokenizer(HarmonyTokenizer):
    chat_template = "{{ messages }}"


class NoTemplateTokenizer(PlainTokenizer):
    chat_template = None

    def get_chat_template(self):
        raise ValueError("No chat template configured")

    def __call__(self, prompt, return_tensors, add_special_tokens=False):
        return super().__call__(prompt, return_tensors, add_special_tokens)


class GenAIHarmonyTokenizer:
    def __init__(self):
        self.applied_history = None
        self.encoded_prompt = None
        self.add_special_tokens = None

    def get_original_chat_template(self):
        return HARMONY_TEMPLATE

    def apply_chat_template(self, history, add_generation_prompt):
        assert add_generation_prompt
        self.applied_history = list(history)
        return "formatted prompt"

    def encode(self, prompt, add_special_tokens=True):
        self.encoded_prompt = prompt
        self.add_special_tokens = add_special_tokens
        return SimpleNamespace(input_ids=SimpleNamespace(data=np.array([[1, 2]], dtype=np.int64)))

    def decode(self, tokens):
        assert tokens == [3]
        return "answer"


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
        if generation_input is not None and not isinstance(generation_input, str):
            return SimpleNamespace(tokens=[[3]])
        return torch.tensor([[1, 2, 3]]) if generation_input is None else "answer"


@pytest.mark.parametrize("tokenizer", [HarmonyTokenizer(), GenAIHarmonyTokenizer()])
def test_detects_harmony_template(tokenizer):
    assert is_harmony_template(tokenizer)
    assert "<|meta_sep|>" in HARMONY_TEMPLATE
    assert "<|channel|>" not in HARMONY_TEMPLATE


@pytest.mark.parametrize("channel_marker", ["<|meta_sep|>", "<|channel|>"])
@pytest.mark.parametrize("tokenizer_api", ["hf", "genai"])
def test_detects_harmony_template_variants(channel_marker, tokenizer_api):
    template = HARMONY_TEMPLATE.replace("<|meta_sep|>", channel_marker)
    if tokenizer_api == "genai":
        tokenizer = SimpleNamespace(get_original_chat_template=lambda: template)
    else:
        tokenizer = SimpleNamespace(chat_template=template)
    assert is_harmony_template(tokenizer)


@pytest.mark.parametrize(
    "template",
    [None, "", {}, {"tool_use": "{{ messages }}"}, {"tool_use": HARMONY_TEMPLATE, "chat": "{{ messages }}"}],
)
def test_missing_hf_template_is_not_harmony(template):
    tokenizer = NoTemplateTokenizer()
    tokenizer.chat_template = template
    assert not is_harmony_template(tokenizer)
    assert not is_native_harmony_prompt("Question", tokenizer)
    assert normalize_chat_history("Question", tokenizer) == [{"role": "user", "content": "Question"}]


@pytest.mark.parametrize("template,expected", [("{{ messages }}", False), (HARMONY_TEMPLATE, True)])
def test_named_hf_templates_resolve_default(template, expected):
    class NamedTemplateTokenizer(PlainTokenizer):
        def get_chat_template(self):
            return self.chat_template["default"]

    tokenizer = NamedTemplateTokenizer()
    tokenizer.chat_template = {"default": template, "tool_use": HARMONY_TEMPLATE}
    assert is_harmony_template(tokenizer) is expected


@pytest.mark.parametrize("strategy_type", [OptimumHFGeneration, KLDivergencyGeneration])
@pytest.mark.parametrize("template", [None, {"tool_use": "{{ messages }}"}])
def test_non_chat_strategy_preserves_tokenizer_default(monkeypatch, strategy_type, template):
    tokenizer = NoTemplateTokenizer()
    tokenizer.chat_template = template
    strategy = strategy_type(metrics_list=[])
    if strategy_type is KLDivergencyGeneration:
        monkeypatch.setattr(strategy, "_score_kl_prompt_tokens_chunked", lambda *_args: torch.zeros((1, 3)))
    strategy.run_generation(FakeModel(), tokenizer, "Question", 4, True, use_chat_template=False)
    assert tokenizer.raw_prompt == "Question"
    assert tokenizer.add_special_tokens is False


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


@pytest.mark.parametrize("separator", ["<|im_sep|>", "\n", "\r\n"])
@pytest.mark.parametrize("include_system", [False, True])
def test_normalizes_chatml_separators(separator, include_system):
    system = f"<|im_start|>system{separator}Rules<|im_end|>" if include_system else ""
    prompt = (
        f"{system}<|im_start|>user{separator}Question<|im_end|>"
        f"<|im_start|>assistant{separator}"
    )
    content = "Rules\r\n\r\nQuestion" if include_system else "Question"
    assert normalize_chat_history(prompt, HarmonyTokenizer()) == [{"role": "user", "content": content}]


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
@pytest.mark.parametrize("backend", ["generation", "kl"])
def test_hf_text_generation_handles_harmony_prompt(
    monkeypatch, prompt, expected_history, expected_raw_prompt, expected_add_special_tokens, backend
):
    monkeypatch.setattr("whowhatbench.text_evaluator.TextSimilarity", lambda *_args: None)
    tokenizer = HarmonyTokenizer()
    if backend == "generation":
        TextEvaluator(
            base_model=FakeModel(),
            tokenizer=tokenizer,
            test_data=[prompt],
            metrics=(),
            use_chat_template=True,
        )
    else:
        strategy = KLDivergencyGeneration(metrics_list=[])
        monkeypatch.setattr(strategy, "_score_kl_prompt_tokens_chunked", lambda *_args: torch.zeros((1, 3)))
        result = strategy.run_generation(FakeModel(), tokenizer, prompt, 4, True, use_chat_template=True)
        assert result.prompt_input_ids.tolist() == [[1, 2]]

    assert tokenizer.applied_history == expected_history
    assert tokenizer.raw_prompt == expected_raw_prompt
    assert tokenizer.add_special_tokens is expected_add_special_tokens


@pytest.mark.parametrize(
    ("prompt", "expected_history", "use_chat_template", "encoded"),
    [
        ("Question", [{"role": "user", "content": "Question"}], True, True),
        ("<|user|>Question<|end|><|assistant|>", [{"role": "user", "content": "Question"}], True, True),
        (
            "<|start|>user<|message|>Question<|end|><|start|>assistant",
            None, True, True,
        ),
        ("<|start|>user<|message|>Question<|end|><|start|>assistant", None, False, True),
        ("Question", None, False, False),
    ],
)
@pytest.mark.parametrize("backend", ["helper", "strategy"])
def test_genai_text_generation_handles_harmony_prompt(
    monkeypatch, prompt, expected_history, use_chat_template, encoded, backend
):
    monkeypatch.setitem(sys.modules, "openvino_genai", SimpleNamespace(ChatHistory=list))
    model = FakeModel(GenAIHarmonyTokenizer())

    if backend == "helper":
        answer = wwb.genai_gen_text(model, None, prompt, 4, True, use_chat_template=use_chat_template)
    else:
        answer = GenAIGeneration(metrics_list=[]).run_generation(
            model, None, prompt, 4, True, use_chat_template=use_chat_template
        ).answer_text
    assert answer == "answer"
    tokenizer = model.get_tokenizer()
    assert tokenizer.applied_history == expected_history
    if encoded:
        assert not isinstance(model.generation_input, (str, list))
        assert tokenizer.encoded_prompt == ("formatted prompt" if expected_history else prompt)
        assert tokenizer.add_special_tokens is False
    else:
        assert model.generation_input == prompt
    assert model.generate_kwargs["apply_chat_template"] is False


@pytest.mark.parametrize(
    ("prompt", "expected_history", "expected_special_tokens"),
    [
        ("Question", [{"role": "user", "content": "Question"}], False),
        ("<|user|>Question<|end|><|assistant|>", [{"role": "user", "content": "Question"}], False),
        ("<|start|>user<|message|>Question<|end|><|start|>assistant", None, False),
    ],
)
def test_genai_token_strategy_handles_harmony_prompt(monkeypatch, prompt, expected_history, expected_special_tokens):
    monkeypatch.setitem(sys.modules, "openvino_genai", SimpleNamespace(ChatHistory=list))
    tokenizer = GenAIHarmonyTokenizer()
    model = FakeModel(tokenizer)
    result = GenAIGeneration(metrics_list=[Metrics.TOKEN_SIMILARITY.value]).run_generation(
        model, None, prompt, 4, True, use_chat_template=True
    )

    assert result.answer_text == "answer"
    assert result.prompt_input_ids.tolist() == [[1, 2]]
    assert result.generated_token_ids.tolist() == [3]
    assert tokenizer.applied_history == expected_history
    assert tokenizer.encoded_prompt == ("formatted prompt" if expected_history else prompt)
    assert tokenizer.add_special_tokens is expected_special_tokens
    assert model.generate_kwargs["apply_chat_template"] is False


def test_genai_token_strategy_preserves_unformatted_input_defaults():
    tokenizer = GenAIHarmonyTokenizer()
    result = GenAIGeneration(metrics_list=[Metrics.TOKEN_SIMILARITY.value]).run_generation(
        FakeModel(tokenizer), None, "Question", 4, True, use_chat_template=False
    )
    assert result.answer_text == "answer"
    assert tokenizer.applied_history is None
    assert tokenizer.encoded_prompt == "Question"
    assert tokenizer.add_special_tokens is True


@pytest.mark.parametrize("backend", ["helper", "strategy"])
@pytest.mark.parametrize("fail_first", [False, True])
def test_independent_genai_prompts_do_not_leave_chat_state(monkeypatch, backend, fail_first):
    monkeypatch.setitem(sys.modules, "openvino_genai", SimpleNamespace(ChatHistory=list))

    class SessionSensitiveModel(FakeModel):
        def __init__(self):
            super().__init__(GenAIHarmonyTokenizer())
            self.chat_active = False
            self.calls = 0

        def generate(self, generation_input=None, **kwargs):
            self.calls += 1
            if isinstance(generation_input, list):
                self.chat_active = True
            elif self.chat_active:
                raise AssertionError("Independent prompt inherited chat mode")
            if fail_first and self.calls == 1:
                raise RuntimeError("Generation failed")
            return super().generate(generation_input, **kwargs)

    model = SessionSensitiveModel()

    def generate(prompt):
        if backend == "helper":
            return wwb.genai_gen_text(model, None, prompt, 4, True, use_chat_template=True)
        return GenAIGeneration(metrics_list=[]).run_generation(
            model, None, prompt, 4, True, use_chat_template=True
        ).answer_text

    if fail_first:
        with pytest.raises(RuntimeError, match="Generation failed"):
            generate("<|user|>Question<|end|><|assistant|>")
    else:
        assert generate("<|user|>Question<|end|><|assistant|>") == "answer"
    assert generate("<|start|>user<|message|>Question<|end|><|start|>assistant") == "answer"
    assert not model.chat_active


def test_identifies_native_harmony_prompt():
    assert is_native_harmony_prompt(
        "<|start|>user<|message|>Question<|end|><|start|>assistant",
        HarmonyTokenizer(),
    )
