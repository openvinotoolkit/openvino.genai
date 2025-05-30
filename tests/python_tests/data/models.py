# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import pytest
from os.path import sep


def get_models_list():
    precommit_models = [
        "katuni4ka/tiny-random-phi3",
    ]

    nightly_models = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "facebook/opt-125m",
        "microsoft/phi-1_5",
        "microsoft/phi-2",
        "THUDM/chatglm3-6b",
        "Qwen/Qwen2-0.5B-Instruct",
        "Qwen/Qwen-7B-Chat",
        "Qwen/Qwen1.5-7B-Chat",
        "argilla/notus-7b-v1",
        "HuggingFaceH4/zephyr-7b-beta",
        "ikala/redpajama-3b-chat",
        "mistralai/Mistral-7B-v0.1",

        # "meta-llama/Llama-2-7b-chat-hf",  # Cannot be downloaded without access token
        # "google/gemma-2b-it",  # Cannot be downloaded without access token.
        # "google/gemma-7b-it",  # Cannot be downloaded without access token.
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "openlm-research/open_llama_3b",
        "openlm-research/open_llama_3b_v2",
        "openlm-research/open_llama_7b",
        "databricks/dolly-v2-12b",
        "databricks/dolly-v2-3b",
    ]

    if pytest.run_marker == "precommit":
        model_ids = precommit_models
    else:
        model_ids = nightly_models

    if pytest.selected_model_ids:
        model_ids = [model_id for model_id in model_ids if model_id in pytest.selected_model_ids.split(' ')]

    return model_ids


def get_chat_models_list():
    precommit_models = [
        "Qwen/Qwen2-0.5B-Instruct",
    ]

    nightly_models = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Llama-2-7b-chat-hf",
        # "google/gemma-2b-it",  # Cannot be downloaded without access token
        # "google/gemma-7b-it",  # Cannot be downloaded without access token
    ]

    if pytest.run_marker == "precommit":
        model_ids = precommit_models
    else:
        model_ids = nightly_models

    return model_ids


def get_gguf_model_list():
    return [
        {
            "hf_model_id": "HuggingFaceTB/SmolLM2-135M",
            "gguf_model_id": "prithivMLmods/SmolLM2-135M-GGUF",
            "gguf_filename": "SmolLM2-135M.F16.gguf"
        },
        {
            "gguf_model_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "gguf_filename": "qwen2.5-0.5b-instruct-q4_0.gguf"
        },
        {
            "gguf_model_id": "Qwen/Qwen3-0.6B-GGUF",
            "gguf_filename": "Qwen3-0.6B-Q8_0.gguf"
        },
    ]
