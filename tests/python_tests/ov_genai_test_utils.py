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
from data.models import get_models_list

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
