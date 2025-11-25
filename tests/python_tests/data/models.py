# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest


def get_models_list() -> tuple[str, ...]:
    model_ids: list[str] = [
        "katuni4ka/tiny-random-phi3",
    ]
    if pytest.selected_model_ids:
        model_ids = [model_id for model_id in model_ids if model_id in pytest.selected_model_ids.split(' ')]

    return tuple(model_ids)


CHAT_MODELS_LIST = (
    "Qwen/Qwen2-0.5B-Instruct",
)


GGUF_MODEL_LIST = (
    {
        "hf_model_id": "HuggingFaceTB/SmolLM2-135M",
        "gguf_model_id": "prithivMLmods/SmolLM2-135M-GGUF",
        "gguf_filename": "SmolLM2-135M.F16.gguf",
        "dynamic_quantization_group_size": None,
    },
    {
        "gguf_model_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "gguf_filename": "qwen2.5-0.5b-instruct-q4_0.gguf",
        "dynamic_quantization_group_size": None,
    },
    pytest.param(
        {
            "hf_model_id": "HuggingFaceTB/SmolLM2-135M",
            "gguf_model_id": "QuantFactory/SmolLM2-135M-GGUF",
            "gguf_filename": "SmolLM2-135M.Q4_1.gguf",
            "dynamic_quantization_group_size": None,
        },
        marks=pytest.mark.xfail(reason="Prediction mismatch. Ticket 172345", raises=AssertionError),
    ),
    {
        "gguf_model_id": "sammysun0711/tiny-random-deepseek-distill-qwen-gguf",
        "gguf_filename": "tiny-random-deepseek-distill-qwen_q8_0.gguf",
        "dynamic_quantization_group_size": "64",
    },
)
