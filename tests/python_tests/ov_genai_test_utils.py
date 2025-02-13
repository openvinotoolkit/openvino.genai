# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pathlib
import os
import pytest
import functools
import openvino
import openvino_tokenizers
import openvino_genai as ov_genai
from typing import List, Tuple
from pathlib import Path
import shutil
import json

import openvino_genai as ov_genai
from common import delete_rt_info

from utils.constants import get_default_llm_properties

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

    prefix = pathlib.Path(os.getenv('GENAI_MODELS_PATH_PREFIX', ''))
    return [(model_id, prefix / model_id.split('/')[1]) for model_id in model_ids]


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

    prefix = pathlib.Path(os.getenv('GENAI_MODELS_PATH_PREFIX', ''))
    return [(model_id, prefix / model_id.split('/')[1]) for model_id in model_ids]


@functools.lru_cache(1)
def read_model(params, **tokenizer_kwargs):
    model_id, models_path = params

    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoTokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if "padding_side" in tokenizer_kwargs:
        hf_tokenizer.padding_side = tokenizer_kwargs.pop("padding_side")

    if (models_path / "openvino_model.xml").exists():
        opt_model = OVModelForCausalLM.from_pretrained(models_path, trust_remote_code=True,
                                                       compile=False, device='CPU', ov_config=get_default_llm_properties())
    else:
        ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(hf_tokenizer,
                                                                             with_detokenizer=True,
                                                                             **tokenizer_kwargs)
        openvino.save_model(ov_tokenizer, models_path / "openvino_tokenizer.xml")
        openvino.save_model(ov_detokenizer, models_path / "openvino_detokenizer.xml")

        # to store tokenizer config jsons with special tokens
        hf_tokenizer.save_pretrained(models_path)

        opt_model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True,
                                                       compile=False, device='CPU', load_in_8bit=False, ov_config=get_default_llm_properties())
        opt_model.generation_config.save_pretrained(models_path)
        opt_model.config.save_pretrained(models_path)
        opt_model.save_pretrained(models_path)

    return (
        model_id,
        models_path,
        hf_tokenizer,
        opt_model,
        ov_genai.LLMPipeline(models_path, 'CPU', ENABLE_MMAP=False, **get_default_llm_properties()),
    )


@pytest.fixture(scope="module")
def model_tmp_path(tmpdir_factory):
    model_id, models_path, _, _, _ = read_model(get_models_list()[0])
    temp_path = tmpdir_factory.mktemp(model_id.replace('/', '_'))

    # copy openvino converted model and tokenizers
    for pattern in ['*.xml', '*.bin']:
        for src_file in models_path.glob(pattern):
            if src_file.is_file():
                shutil.copy(src_file, temp_path / src_file.name)

    yield model_id, Path(temp_path)


def load_genai_pipe_with_configs(configs: List[Tuple], temp_path):
    # Load LLMPipeline where all configs are cleared.
    # remove existing jsons from previous tests
    for json_file in temp_path.glob("*.json"):
        json_file.unlink()
    delete_rt_info(configs, temp_path)

    for config_json, config_name in configs:
        with (temp_path / config_name).open('w') as f:
            json.dump(config_json, f)

    ov_pipe = ov_genai.LLMPipeline(temp_path, 'CPU', **get_default_llm_properties())

    for _, config_name in configs:
        os.remove(temp_path / config_name)

    return ov_pipe


@functools.lru_cache(1)
def get_continuous_batching(path):
    return ov_genai.LLMPipeline(path, 'CPU', scheduler_config=ov_genai.SchedulerConfig(), **get_default_llm_properties())
