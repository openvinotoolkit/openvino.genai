# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, List
from openvino_genai import GenerationConfig
from utils.generation_config import get_greedy, get_beam_search

def get_test_dataset() -> Tuple[List[str], List[GenerationConfig]]:
    prompts = [
        "What is OpenVINO?",
        "How are you?",
        "What is your name?",
        "Tell me something about Canada"
    ]
    generation_configs = [
        get_greedy(),
        get_beam_search(),
        get_greedy(),
        get_beam_search(),
    ]
    return (prompts, generation_configs)