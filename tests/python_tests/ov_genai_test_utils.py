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


def get_models_list():
    precommit_models = [
        "katuni4ka/tiny-random-phi3",
    ]

    nightly_models = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "facebook/opt-125m",
        "microsoft/phi-1_5",
        "microsoft/phi-2",
        "THUDM/chatglm2-6b",
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
    # pytest.set_trace()
    prefix = pathlib.Path(os.getenv('GENAI_MODELS_PATH_PREFIX', ''))
    return [(model_id, prefix / model_id.split('/')[1]) for model_id in model_ids]


def get_whisper_models_list(tiny_only=False, multilingual=False, en_only=False):
    precommit_models = [
        "openai/whisper-tiny",
        "openai/whisper-tiny.en",
        "distil-whisper/distil-small.en",
    ]
    if multilingual:
        precommit_models = ["openai/whisper-tiny"]
    if en_only:
        precommit_models = ["openai/whisper-tiny.en", "distil-whisper/distil-small.en"]
    if tiny_only:
        precommit_models = ["openai/whisper-tiny"]

    nightly_models = []

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


def get_chat_templates():
    # Returns chat templates saved in tokenizer_configs.py, 
    # but skips some models that currently are not processed correctly.

    skipped_models = {
        # TODO: openchat/openchat_3.5 and berkeley-nest/Starling-LM-7B-alpha have the same template.
        # Need to enable and unskip, since it's preset in continious batching and has >100 000 downloads.
        "openchat/openchat-3.5-0106",
        
        # These models fail even on HF so no need to check if applying chat matches.
        "vibhorag101/llama-2-13b-chat-hf-phr_mental_therapy",
        "codellama/CodeLlama-34b-Instruct-hf",
        "deepseek-ai/deepseek-math-7b-rl",
        "allenai/tulu-2-7b",
        "alexsobolev/IcaroLM",
        "tokyotech-llm/Swallow-7b-instruct-v0.1",
        "bofenghuang/vigogne-2-7b-chat",
        "OpenBuddy/openbuddy-mistral2-7b-v20.3-32k",
        "AliAbdelrasheed/maqa_llama_4bit",
        "stephenlzc/Mistral-7B-v0.3-Chinese-Chat-uncensored",

        # TODO: Need to support chat templates in more models: CVS-145963
        # Either ov_genai is unable to parse chat_template or results do not match with HF.
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "databricks/dbrx-instruct", # Chat template is not supported by Jinja2Cpp
        "mosaicml/mpt-30b-chat",
        "deepseek-ai/deepseek-coder-6.7b-instruct", # Chat template is not supported by Jinja2Cpp
        "maldv/winter-garden-7b-alpha", # Chat template is not supported by Jinja2Cpp
        "ishorn5/RTLCoder-Deepseek-v1.1", # Chat template is not supported by Jinja2Cpp
        "openchat/openchat-3.5-0106",
        "casperhansen/llama-3-70b-instruct-awq",
        "TheBloke/deepseek-coder-33B-instruct-GPTQ",
        "AI-Sweden-Models/gpt-sw3-356m-instruct",
        "google/gemma-7b-it",
        "THUDM/cogvlm2-llama3-chat-19B",
        "KnutJaegersberg/internlm-20b-llama",
        "maywell/Synatra-Mixtral-8x7B",
        "MediaTek-Research/Breeze-7B-Instruct-v1_0",
        "bofenghuang/vigostral-7b-chat",
        "meetkai/functionary-small-v2.5", # Chat template is not supported by Jinja2Cpp
        "openchat/openchat-3.6-8b-20240522",
        "tenyx/TenyxChat-7B-v1",
        "LoneStriker/TinyLlama-1.1B-32k-Instruct-3.0bpw-h6-exl2",
        "yam-peleg/Hebrew-Gemma-11B-V2",
        "shenzhi-wang/Llama3-8B-Chinese-Chat", # AssertionError
        "nlpai-lab/KULLM3",
        "HuggingFaceH4/zephyr-7b-gemma-sft-v0.1",
        "MediaTek-Research/Breeze-7B-Instruct-v0_1", 
        "shanchen/llama3-8B-slerp-biomed-chat-chinese", # AssertionError
        "MLP-KTLim/llama-3-Korean-Bllossom-8B",
        "aloobun/CosmicBun-8B", # Chat template is not supported by Jinja2Cpp
        "codellama/CodeLlama-70b-Instruct-hf",
        "gorilla-llm/gorilla-openfunctions-v2", # Chat template is not supported by Jinja2Cpp
        "BramVanroy/Llama-2-13b-chat-dutch"
    }
    from tokenizer_configs import get_tokenizer_configs
    return [(k, v) for k, v in get_tokenizer_configs().items() if k not in skipped_models]


@functools.lru_cache(1)
def read_model(params, **tokenizer_kwargs):
    model_id, path = params
    
    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if (path / "openvino_model.xml").exists():
        opt_model = OVModelForCausalLM.from_pretrained(path, trust_remote_code=True, 
                                                       compile=False, device='CPU')
    else:
        ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(tokenizer, 
                                                                             with_detokenizer=True,
                                                                             **tokenizer_kwargs)
        openvino.save_model(ov_tokenizer, path / "openvino_tokenizer.xml")
        openvino.save_model(ov_detokenizer, path / "openvino_detokenizer.xml")
        
        # to store tokenizer config jsons with special tokens
        tokenizer.save_pretrained(path)
        
        opt_model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True, 
                                                       compile=False, device='CPU', load_in_8bit=False)
        opt_model.generation_config.save_pretrained(path)
        opt_model.config.save_pretrained(path)
        opt_model.save_pretrained(path)
    
    return (
        model_id,
        path,
        tokenizer,
        opt_model,
        ov_genai.LLMPipeline(str(path), device='CPU', **{"ENABLE_MMAP": False}),
    )


# in OpenVINO GenAI this parameter is called stop_criteria,
# while in HF it's called early_stopping. 
# HF values True, False and "never" correspond to OV GenAI values "EARLY", "HEURISTIC" and "NEVER"
STOP_CRITERIA_MAP = {
    ov_genai.StopCriteria.NEVER: "never", 
    ov_genai.StopCriteria.EARLY: True, 
    ov_genai.StopCriteria.HEURISTIC: False
}


@pytest.fixture(scope="module")
def model_tmp_path(tmpdir_factory):
    model_id, path, _, _, _ = read_model(get_models_list()[0])
    temp_path = tmpdir_factory.mktemp(model_id.replace('/', '_'))

    # copy openvino converted model and tokenizers
    for pattern in ['*.xml', '*.bin']:
        for src_file in path.glob(pattern):
            if src_file.is_file():
                shutil.copy(src_file, temp_path / src_file.name)    
    yield model_id, Path(temp_path)


def load_tok(configs: List[Tuple], temp_path):
    # load Tokenizer where all configs are cleared.
    # remove existing jsons from previous tests
    for json_file in temp_path.glob("*.json"):
        json_file.unlink()

    for config_json, config_name in configs:
        with (temp_path / config_name).open('w') as f:
            json.dump(config_json, f)
    return ov_genai.Tokenizer(str(temp_path), {})


def load_pipe(configs: List[Tuple], temp_path):
    # Load LLMPipline where all configs are cleared.
    # remove existing jsons from previous tests
    for json_file in temp_path.glob("*.json"):
        json_file.unlink()

    for config_json, config_name in configs:
        with (temp_path / config_name).open('w') as f:
            json.dump(config_json, f)
    return ov_genai.LLMPipeline(str(temp_path))


@functools.lru_cache(1)
def get_continuous_batching(path):
    scheduler_config = ov_genai.SchedulerConfig()
    scheduler_config.cache_size = 1
    return ov_genai.LLMPipeline(str(path), ov_genai.Tokenizer(str(path)), device='CPU', **{"scheduler_config": scheduler_config})
