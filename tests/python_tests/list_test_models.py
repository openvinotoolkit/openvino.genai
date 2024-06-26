# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pathlib
import os

def models_list():
    model_ids = [
        "katuni4ka/tiny-random-phi3",
        # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # "facebook/opt-125m",

        # "microsoft/phi-1_5",
        # "microsoft/phi-2",
        # "THUDM/chatglm2-6b",
        # "Qwen/Qwen2-0.5B-Instruct",
        # "Qwen/Qwen-7B-Chat",
        # "Qwen/Qwen1.5-7B-Chat",
        # "argilla/notus-7b-v1",
        # "HuggingFaceH4/zephyr-7b-beta",
        # "ikala/redpajama-3b-chat",
        # "mistralai/Mistral-7B-v0.1",
        
        # "meta-llama/Llama-2-7b-chat-hf",
        # "google/gemma-2b-it",
        # "meta-llama/Llama-2-13b-chat-hf",
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        # "openlm-research/open_llama_3b",
        # "openlm-research/open_llama_3b_v2",
        # "openlm-research/open_llama_7b",
        # "databricks/dolly-v2-12b",
        # "databricks/dolly-v2-3b",
    ]

    prefix = pathlib.Path(os.getenv('GENAI_MODELS_PATH_PREFIX', ''))
    return [(model_id, prefix / model_id.split('/')[1]) for model_id in model_ids]


def chat_models_list():
    model_ids = [
        "mosaicml/mpt-7b-chat",
        # "Qwen/Qwen2-0.5B-Instruct",
        # "Qwen/Qwen2-1.5B-Instruct",
        # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        # "meta-llama/Llama-2-7b-chat-hf",
        # "google/gemma-2b-it",
        # "google/gemma-7b-it",
    ]

    prefix = pathlib.Path(os.getenv('GENAI_MODELS_PATH_PREFIX', ''))
    return [(model_id, prefix / model_id.split('/')[1]) for model_id in model_ids]


if __name__ == "__main__":
    for model_id, model_path in models_list():
        print(model_id, model_path)
