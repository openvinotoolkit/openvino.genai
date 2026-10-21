# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

LLM_BENCH_DIR = Path(__file__).parents[2] / "tools" / "llm_bench"
sys.path.insert(0, str(LLM_BENCH_DIR))

from task.text_generation import (
    _is_harmony_template,
    _should_apply_chat_template,
    apply_chat_template_genai,
    run_text_generation_genai,
    run_text_generation_genai_with_stream,
)


HARMONY_TEMPLATE = (
    "{{- '<|start|>' + message['role'] -}}"
    "{% if message['channel'] is defined %}{{- '<|meta_sep|>' + message['channel'] -}}{% endif %}"
    "{{- '<|message|>' + message['content'] + '<|end|>' -}}"
)


class HarmonyTokenizer:
    def get_original_chat_template(self):
        return HARMONY_TEMPLATE

    def apply_chat_template(self, history, add_generation_prompt):
        messages = "|".join(f"{message['role']}:{message['content']}" for message in history)
        return f"<|start|>{messages}<|start|>assistant"


class PlainTokenizer:
    def get_original_chat_template(self):
        return "<|user|>{{ content }}"

    def apply_chat_template(self, history, add_generation_prompt):
        messages = "|".join(f"{message['role']}:{message['content']}" for message in history)
        return f"<|start|>{messages}<|start|>assistant"


@pytest.mark.parametrize("channel_marker", ["<|meta_sep|>", "<|channel|>"])
def test_detects_representative_harmony_template(channel_marker):
    template = HARMONY_TEMPLATE.replace("<|meta_sep|>", channel_marker)
    assert _is_harmony_template(SimpleNamespace(get_original_chat_template=lambda: template))
    assert "<|meta_sep|>" in HARMONY_TEMPLATE
    assert "<|channel|>" not in HARMONY_TEMPLATE


def test_harmony_template_converts_legacy_user_prompt():
    args = {"apply_chat_template": False, "batch_size": 1, "disable_prompt_permutation": True}
    prompt = "<|user|>Summarize this article.<|end|><|assistant|>"

    assert _should_apply_chat_template(args, prompt, HarmonyTokenizer())
    assert apply_chat_template_genai(args, prompt, HarmonyTokenizer()) == [
        "<|start|>user:Summarize this article.<|start|>assistant"
    ]


@pytest.mark.parametrize("separator", ["<|im_sep|>", "\n", "\r\n"])
@pytest.mark.parametrize("include_system", [False, True])
def test_harmony_template_converts_legacy_chatml_prompt(separator, include_system):
    args = {"apply_chat_template": False, "batch_size": 1, "disable_prompt_permutation": True}
    system = f"<|im_start|>system{separator}Summarize precisely.<|im_end|>" if include_system else ""
    prompt = (
        f"{system}<|im_start|>user{separator}Article text.<|im_end|>"
        f"<|im_start|>assistant{separator}"
    )
    content = "Summarize precisely.\r\n\r\nArticle text." if include_system else "Article text."

    assert apply_chat_template_genai(args, prompt, HarmonyTokenizer()) == [
        f"<|start|>user:{content}<|start|>assistant"
    ]


@pytest.mark.parametrize("explicit_template", [False, True])
def test_harmony_template_does_not_wrap_native_prompt(explicit_template):
    args = {"apply_chat_template": explicit_template, "batch_size": 1, "disable_prompt_permutation": True}
    prompt = "<|start|>user<|message|>Article text.<|end|><|start|>assistant"

    assert not _should_apply_chat_template(args, prompt, HarmonyTokenizer())
    assert apply_chat_template_genai(args, prompt, HarmonyTokenizer()) == [prompt]


def test_plain_template_keeps_existing_default_behavior():
    args = {"apply_chat_template": False, "batch_size": 1, "disable_prompt_permutation": True}

    assert not _should_apply_chat_template(args, "Article text.", PlainTokenizer())


def test_plain_template_preserves_legacy_prompt_with_explicit_flag():
    args = {"apply_chat_template": True, "batch_size": 1, "disable_prompt_permutation": True}
    prompt = "<|user|>Article text.<|end|><|assistant|>"

    assert apply_chat_template_genai(args, prompt, PlainTokenizer()) == [
        "<|start|>user:<|user|>Article text.<|end|><|assistant|><|start|>assistant"
    ]


@pytest.mark.parametrize("tokenizer", [SimpleNamespace(), SimpleNamespace(get_original_chat_template=lambda: None)])
def test_missing_chat_template_is_not_harmony(tokenizer):
    assert not _is_harmony_template(tokenizer)
    assert not _should_apply_chat_template({"apply_chat_template": False}, "Question", tokenizer)


@pytest.mark.parametrize("runner", [run_text_generation_genai, run_text_generation_genai_with_stream])
@pytest.mark.parametrize(
    ("template", "prompt", "explicit_template", "expected_prompt", "expected_kwargs"),
    [
        (HARMONY_TEMPLATE, "Question", False, "<|start|>user:Question<|start|>assistant", {"add_special_tokens": False}),
        (
            HARMONY_TEMPLATE, "<|user|>Question<|end|><|assistant|>", False,
            "<|start|>user:Question<|start|>assistant", {"add_special_tokens": False},
        ),
        (
            HARMONY_TEMPLATE, "<|start|>user<|message|>Question<|end|><|start|>assistant", True,
            "<|start|>user<|message|>Question<|end|><|start|>assistant", {"add_special_tokens": False},
        ),
        (
            HARMONY_TEMPLATE, "<|start|>user<|message|>Question<|end|><|start|>assistant", False,
            "<|start|>user<|message|>Question<|end|><|start|>assistant", {"add_special_tokens": False},
        ),
        ("{{ messages }}", "Question", False, "Question", {}),
        ("{{ messages }}", "Question", True, "<|start|>user:Question<|start|>assistant", {}),
        (None, "Question", False, "Question", {}),
    ],
)
def test_genai_runners_share_prompt_preparation(
    runner, template, prompt, explicit_template, expected_prompt, expected_kwargs
):
    class TokenizationComplete(Exception):
        pass

    class RecordingTokenizer(HarmonyTokenizer):
        def get_original_chat_template(self):
            return template

        def encode(self, prompts, **kwargs):
            self.encoded_prompts = prompts
            self.encode_kwargs = kwargs
            raise TokenizationComplete

    tokenizer = RecordingTokenizer()
    args = {
        "apply_chat_template": explicit_template, "batch_size": 1,
        "disable_prompt_permutation": True, "output_dir": None,
    }
    with pytest.raises(TokenizationComplete):
        runner(
            input_text=prompt, num=0, model=SimpleNamespace(get_tokenizer=lambda: tokenizer), tokenizer=tokenizer,
            args=args, iter_data_list=[], md5_list=[{}], prompt_index=0, streamer=None, tokens_len=None,
            streaming=False, model_precision="FP32", proc_id=0, mem_consumption=None, prefix="test",
        )

    assert tokenizer.encoded_prompts == [expected_prompt]
    assert tokenizer.encode_kwargs == expected_kwargs
