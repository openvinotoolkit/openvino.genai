# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino_genai import GenerationConfig
from utils.generation_config import get_greedy, get_beam_search
from optimum.intel.utils.import_utils import is_transformers_version

def get_test_dataset() -> tuple[list[str], list[GenerationConfig]]:
    # separated due to group beam search fails with optimum-intel 423b423 and transformers>=5.0
    # restore after fix of CVS-185790
    if is_transformers_version("<", "5.0"):
        generation_configs = [
            get_beam_search(),
            get_beam_search(),
        ]

        prompts = ["How are you?", "Tell me something about Canada"]
    else:
        generation_configs = [
            get_greedy(),
            get_greedy(),
        ]

        prompts = [
            "What is OpenVINO?",
            "What is your name?",
        ]

    return (prompts, generation_configs)
