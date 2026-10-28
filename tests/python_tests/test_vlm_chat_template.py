# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
from openvino_genai import ContinuousBatchingPipeline, SchedulerConfig, VLMPipeline

from test_vlm_pipeline import GEMMA3_MACOS_XFAIL_REASON, MODEL_GEMMA, _get_ov_model


# (model_id, raw template, message content, expected rendering after the model patch)
PATCHED_CHAT_TEMPLATE_CASES = [
    pytest.param(
        MODEL_GEMMA,
        "[{{ messages[0]['content'] | trim }}]",
        "  hi  ",
        "[  hi  ]",
        id="gemma3-trim-removed",
    ),
    pytest.param(
        "optimum-intel-internal-testing/tiny-random-gemma4",
        '{{ "first "\n    "second" }}',
        "unused",
        "first second",
        id="gemma4-multiline-literals",
    ),
    pytest.param(
        "optimum-intel-internal-testing/tiny-random-qwen3.5",
        "{% if enable_thinking is undefined %}default{% else %}explicit{% endif %}",
        "unused",
        "default",
        id="qwen3.5-is-undefined",
    ),
]


def _create_pipeline(pipeline_kind: str, model_id: str):
    if sys.platform == "darwin" and "gemma3" in model_id:
        pytest.xfail(GEMMA3_MACOS_XFAIL_REASON)
    models_path = _get_ov_model(model_id)
    if pipeline_kind == "vlm":
        return VLMPipeline(models_path, "CPU")
    return ContinuousBatchingPipeline(models_path, SchedulerConfig(), "CPU")


@pytest.mark.parametrize("pipeline_kind", ["vlm", "continuous_batching"])
@pytest.mark.parametrize("model_id, raw_template, content, expected", PATCHED_CHAT_TEMPLATE_CASES)
def test_set_chat_template_applies_model_patch(pipeline_kind, model_id, raw_template, content, expected):
    pipeline = _create_pipeline(pipeline_kind, model_id)
    tokenizer = pipeline.get_tokenizer()
    history = [{"role": "user", "content": content}]

    tokenizer.set_chat_template(raw_template)

    assert tokenizer.apply_chat_template(history, add_generation_prompt=False) == expected
    assert tokenizer.get_original_chat_template() == raw_template
    assert tokenizer.chat_template != raw_template


@pytest.mark.parametrize("pipeline_kind", ["vlm", "continuous_batching"])
@pytest.mark.parametrize("model_id, raw_template, content, expected", PATCHED_CHAT_TEMPLATE_CASES)
def test_inline_chat_template_is_patched_and_not_stored(pipeline_kind, model_id, raw_template, content, expected):
    pipeline = _create_pipeline(pipeline_kind, model_id)
    tokenizer = pipeline.get_tokenizer()
    history = [{"role": "user", "content": content}]
    stored_template = tokenizer.chat_template
    original_template = tokenizer.get_original_chat_template()

    rendered = tokenizer.apply_chat_template(history, add_generation_prompt=False, chat_template=raw_template)

    assert rendered == expected
    assert tokenizer.chat_template == stored_template
    assert tokenizer.get_original_chat_template() == original_template
