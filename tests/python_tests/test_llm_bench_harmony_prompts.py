# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

from openvino_genai import GenerationConfig

LLM_BENCH_DIR = Path(__file__).parents[2] / "tools" / "llm_bench"
sys.path.insert(0, str(LLM_BENCH_DIR))

from task.text_generation import _should_apply_chat_template, apply_chat_template_genai, genai_generation_config_setup


class HarmonyTokenizer:
    def get_original_chat_template(self):
        return "<|start|>assistant<|channel|>final<|message|>"

    def apply_chat_template(self, history, add_generation_prompt):
        messages = "|".join(f"{message['role']}:{message['content']}" for message in history)
        return f"<|start|>{messages}<|start|>assistant"


class PlainTokenizer:
    def get_original_chat_template(self):
        return "<|user|>{{ content }}"

    def apply_chat_template(self, history, add_generation_prompt):
        messages = "|".join(f"{message['role']}:{message['content']}" for message in history)
        return f"<|start|>{messages}<|start|>assistant"


class GenerationConfigModel:
    def get_generation_config(self):
        return GenerationConfig()


def get_generation_args(end_token_stopping):
    return {
        "end_token_stopping": end_token_stopping,
        "seed": 42,
        "num_beams": 1,
        "draft_model": "",
        "max_ngram_size": None,
        "num_assistant_tokens": None,
    }


def test_harmony_template_converts_legacy_user_prompt():
    args = {"apply_chat_template": False, "batch_size": 1, "disable_prompt_permutation": True}
    prompt = "<|user|>Summarize this article.<|end|><|assistant|>"

    assert _should_apply_chat_template(args, prompt, HarmonyTokenizer())
    assert apply_chat_template_genai(args, prompt, HarmonyTokenizer()) == [
        "<|start|>user:Summarize this article.<|start|>assistant"
    ]


def test_harmony_template_converts_legacy_chatml_prompt():
    args = {"apply_chat_template": False, "batch_size": 1, "disable_prompt_permutation": True}
    prompt = (
        "<|im_start|>system<|im_sep|>Summarize precisely.<|im_end|>"
        "<|im_start|>user<|im_sep|>Article text.<|im_end|>"
        "<|im_start|>assistant<|im_sep|>"
    )

    assert apply_chat_template_genai(args, prompt, HarmonyTokenizer()) == [
        "<|start|>user:Summarize precisely.\r\n\r\nArticle text.<|start|>assistant"
    ]


def test_harmony_template_does_not_wrap_native_prompt():
    args = {"apply_chat_template": False, "batch_size": 1, "disable_prompt_permutation": True}
    prompt = "<|start|>user<|message|>Article text.<|end|><|start|>assistant"

    assert not _should_apply_chat_template(args, prompt, HarmonyTokenizer())


def test_plain_template_keeps_existing_default_behavior():
    args = {"apply_chat_template": False, "batch_size": 1, "disable_prompt_permutation": True}

    assert not _should_apply_chat_template(args, "Article text.", PlainTokenizer())


def test_plain_template_preserves_legacy_prompt_with_explicit_flag():
    args = {"apply_chat_template": True, "batch_size": 1, "disable_prompt_permutation": True}
    prompt = "<|user|>Article text.<|end|><|assistant|>"

    assert apply_chat_template_genai(args, prompt, PlainTokenizer()) == [
        "<|start|>user:<|user|>Article text.<|end|><|assistant|><|start|>assistant"
    ]


def test_genai_honors_end_token_stopping():
    generation_config = genai_generation_config_setup(GenerationConfigModel(), 128, get_generation_args(True))

    assert not generation_config.ignore_eos


def test_genai_keeps_fixed_token_benchmark_default():
    generation_config = genai_generation_config_setup(GenerationConfigModel(), 128, get_generation_args(False))

    assert generation_config.ignore_eos