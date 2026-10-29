# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from optimum.utils.import_utils import is_transformers_version

from test_vlm_pipeline import MODEL_GEMMA, ov_continuous_batching_pipe, ov_pipe_model


GEMMA3_TEMPLATE = "[{{ messages[0]['content'] | trim }}]"
GEMMA3_REVISION = "0cc4c5a822bbc68aeb312fd789046463ca301d14"
GEMMA4_TEMPLATE = '{{ "first "\n    "second" }}'
GEMMA4_REVISION = "422d12880b72b92b1547ac7ce2e1a4865f6bec6c"
QWEN35_TEMPLATE = "{% if enable_thinking is undefined %}default{% else %}explicit{% endif %}"
QWEN35_REVISION = "aa8ac4a953c6081e0afaa34bd95cefd9c852ddbf"
QWEN35_MODEL = "optimum-intel-internal-testing/tiny-random-qwen3.5"

GEMMA3_LEGACY_TRANSFORMERS = [
    pytest.mark.transformers_lower_v5,
    pytest.mark.skipif(
        is_transformers_version(">=", "5.0"),
        reason="Gemma 3 tiny-model export is unsupported with Transformers >=5",
    ),
]
QWEN35_TRANSFORMERS = pytest.mark.skipif(
    not (is_transformers_version(">=", "5.2.0") and is_transformers_version("<", "5.3.0")),
    reason="Qwen 3.5 tiny-model export is covered by the Transformers 5.2.x VLM job",
)


VLM_TEMPLATE_CASES = (
    [
        pytest.param(
            (MODEL_GEMMA, backend, False, GEMMA3_REVISION),
            GEMMA3_TEMPLATE,
            "  hi  ",
            "[  hi  ]",
            marks=GEMMA3_LEGACY_TRANSFORMERS,
            id=f"gemma3-trim-{backend.lower()}",
        )
        for backend in ("PA", "SDPA")
    ]
    + [
        pytest.param(
            ("optimum-intel-internal-testing/tiny-random-gemma4", backend, False, GEMMA4_REVISION),
            GEMMA4_TEMPLATE,
            "unused",
            "first second",
            id=f"gemma4-multiline-{backend.lower()}",
        )
        for backend in ("PA", "SDPA")
    ]
    + [
        pytest.param(
            (QWEN35_MODEL, backend, False, QWEN35_REVISION),
            QWEN35_TEMPLATE,
            "unused",
            "default",
            marks=QWEN35_TRANSFORMERS,
            id=f"qwen3.5-undefined-{backend.lower()}",
        )
        for backend in ("PA", "SDPA")
    ]
)

CB_TEMPLATE_CASES = [
    pytest.param(
        (MODEL_GEMMA, GEMMA3_REVISION),
        GEMMA3_TEMPLATE,
        "  hi  ",
        "[  hi  ]",
        marks=GEMMA3_LEGACY_TRANSFORMERS,
        id="gemma3-trim",
    ),
    pytest.param(
        (QWEN35_MODEL, QWEN35_REVISION),
        QWEN35_TEMPLATE,
        "unused",
        "default",
        marks=QWEN35_TRANSFORMERS,
        id="qwen3.5-undefined",
    ),
]


def _assert_set_chat_template(tokenizer, raw_template, content, expected):
    original_template = tokenizer.get_original_chat_template()
    tokenizer.set_chat_template(raw_template)
    history = [{"role": "user", "content": content}]

    try:
        assert tokenizer.apply_chat_template(history, add_generation_prompt=False) == expected
        assert tokenizer.get_original_chat_template() == raw_template
        assert tokenizer.chat_template != raw_template
    finally:
        tokenizer.set_chat_template(original_template)


def _assert_inline_chat_template(tokenizer, raw_template, content, expected):
    history = [{"role": "user", "content": content}]
    stored_template = tokenizer.chat_template
    original_template = tokenizer.get_original_chat_template()

    rendered = tokenizer.apply_chat_template(history, add_generation_prompt=False, chat_template=raw_template)

    assert rendered == expected
    assert tokenizer.chat_template == stored_template
    assert tokenizer.get_original_chat_template() == original_template


@pytest.mark.parametrize(
    "ov_pipe_model, raw_template, content, expected", VLM_TEMPLATE_CASES, indirect=["ov_pipe_model"]
)
def test_set_chat_template_applies_model_patch(ov_pipe_model, raw_template, content, expected):
    _assert_set_chat_template(ov_pipe_model.pipeline.get_tokenizer(), raw_template, content, expected)


@pytest.mark.parametrize(
    "ov_pipe_model, raw_template, content, expected", VLM_TEMPLATE_CASES, indirect=["ov_pipe_model"]
)
def test_inline_chat_template_is_patched_and_not_stored(ov_pipe_model, raw_template, content, expected):
    _assert_inline_chat_template(ov_pipe_model.pipeline.get_tokenizer(), raw_template, content, expected)


@pytest.mark.parametrize(
    "ov_continuous_batching_pipe, raw_template, content, expected",
    CB_TEMPLATE_CASES,
    indirect=["ov_continuous_batching_pipe"],
)
def test_cb_set_chat_template_applies_model_patch(ov_continuous_batching_pipe, raw_template, content, expected):
    _assert_set_chat_template(ov_continuous_batching_pipe.get_tokenizer(), raw_template, content, expected)


@pytest.mark.parametrize(
    "ov_continuous_batching_pipe, raw_template, content, expected",
    CB_TEMPLATE_CASES,
    indirect=["ov_continuous_batching_pipe"],
)
def test_cb_inline_chat_template_is_patched_and_not_stored(
    ov_continuous_batching_pipe, raw_template, content, expected
):
    _assert_inline_chat_template(ov_continuous_batching_pipe.get_tokenizer(), raw_template, content, expected)
