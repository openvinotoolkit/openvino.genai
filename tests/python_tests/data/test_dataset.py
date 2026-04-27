# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino_genai import GenerationConfig
from utils.generation_config import get_greedy, get_beam_search
from optimum.intel.utils.import_utils import is_transformers_version

def get_test_dataset() -> tuple[list[str], list[GenerationConfig]]:
    prompts = [
        "What is OpenVINO?",
        "How are you?",
        "What is your name?",
        "Tell me something about Canada"
    ]

    if is_transformers_version(">=", "5.0"):
        generation_configs = [
            get_beam_search(),
            get_beam_search(),
            get_beam_search(),
            get_beam_search(),
        ]
    else:
        generation_configs = [
            get_greedy(),
            get_greedy(),
            get_greedy(),
            get_greedy(),
        ]

    return (prompts, generation_configs)
