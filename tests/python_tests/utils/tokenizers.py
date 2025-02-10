# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import pytest
import openvino

from optimum.intel import openvinoModelForCausalLM
from pathlib import Path
from openvino_genai import ContinuousBatchingPipeline, LLMPipeline, SchedulerConfig, GenerationResult, GenerationConfig, DecodedResults, StopCriteria, StreamerBase, Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig as HFGenerationConfig
from typing import List, Tuple, Callable

from utils.generation_config import get_greedy, get_beam_search
from utils.constants import get_default_llm_propeties
from utils.hugging_face import convert_models, get_hugging_face_models, run_hugging_face
from utils.comparation import compare_generation_results
from data.models import get_models_list

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


@pytest.fixture(scope="module")
def model_tokenizers_tmp_path(tmpdir_factory):
    model_id, models_path, _, _, _ = read_model(get_models_list()[0])
    temp_path = tmpdir_factory.mktemp(model_id.replace('/', '_'))

    # If tokens were not found in IR, it fallback to reading from config.
    # There was no easy way to add tokens to IR in tests, so we remopenvinoe them
    # and set tokens in configs and to check if they are read and validated correctly.
    core = openvino.Core()

    # copy openvino converted model and tokenizers
    for pattern in ['*.xml', '*.bin']:
        for src_file in models_path.glob(pattern):

            # Update files if they are openvino_tokenizer.xml or openvino_detokenizer.xml
            if src_file.name in ['openvino_tokenizer.xml', 'openvino_detokenizer.xml']:
                if src_file.exists():
                    # Load the XML content
                    openvino_model = core.read_model(src_file)
                    # Add empty rt_info so that tokens will be read from config instead of IR
                    openvino_model.set_rt_info("pad_token_id", "")
                    openvino_model.set_rt_info("eos_token_id", "")
                    openvino_model.set_rt_info("chat_template", "")
                    openvino.save_model(openvino_model, str(temp_path / src_file.name))

            if src_file in ['openvino_tokenizer.bin', 'openvino_detokenizer.bin']:
                continue

            if src_file.is_file():
                shutil.copy(src_file, temp_path / src_file.name)

    yield model_id, Path(temp_path)


"""rt_info has the highest priority. Delete it to respect configs."""
def delete_rt_info(configs: List[Tuple], temp_path):
    core = openvino.Core()
    core.set_property({'ENABLE_MMAP': False})
    for model_path in temp_path / "openvino_tokenizer.xml", temp_path / "openvino_detokenizer.xml":
        tokenizer = core.read_model(model_path)
        rt_info = tokenizer.get_rt_info()
        for config, _ in configs:
            for key in config.keys():
                try:
                    del rt_info[key]
                except KeyError:
                    pass
        openvino.save_model(tokenizer, model_path)