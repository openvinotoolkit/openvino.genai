# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino_genai import GenerationConfig
from utils.generation_config import get_greedy, get_beam_search

PROMPTS = [
    "What is OpenVINO?",
    "How are you?",
    "What is your name?",
    "Tell me something about Canada"
]

GENERATION_CONFIGS = [
    get_greedy(),
    get_beam_search(),
    get_greedy(),
    get_beam_search(),
]

def get_test_dataset() -> tuple[list[str], list[GenerationConfig]]:
    return PROMPTS, GENERATION_CONFIGS
