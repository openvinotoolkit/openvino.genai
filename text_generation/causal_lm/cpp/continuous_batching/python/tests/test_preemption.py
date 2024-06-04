# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from dataclasses import dataclass
from py_continuous_batching import GenerationConfig, GenerationResult
from typing import List

from common import get_model_and_tokenizer, save_ov_model_from_optimum, generate_and_compare_with_reference_text, \
    DEFAULT_SCHEDULER_CONFIG, get_scheduler_config, run_test_pipeline, get_models_list, get_beam_search, get_greedy, \
    get_multinomial_temperature_and_top_k, get_multinomial_temperature, get_multinomial_temperature_and_top_p, \
    get_multinomial_temperature_and_num_return_sequence, get_multinomial_all_parameters
from test_sampling import RandomSamplingTestStruct

scheduler_params_list = [({"num_kv_blocks": 2, "block_size": 32, "dynamic_split_fuse": True, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_greedy()),
                         ({"num_kv_blocks": 2, "block_size": 32, "dynamic_split_fuse": False, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_greedy()),
                         ({"num_kv_blocks": 34, "block_size": 32, "dynamic_split_fuse": True, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_beam_search()),
                         ({"num_kv_blocks": 34, "block_size": 32, "dynamic_split_fuse": False, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_beam_search())]
@pytest.mark.parametrize("params", scheduler_params_list)
@pytest.mark.precommit
def test_preemption(tmp_path, params):
    run_test_pipeline(tmp_path, "facebook/opt-125m", params[0], params[1])

multinomial_params = RandomSamplingTestStruct(generation_config=[
                                                          get_multinomial_temperature(),
                                                          get_multinomial_temperature_and_top_p(),
                                                          get_multinomial_temperature_and_top_k(),
                                                          get_multinomial_temperature_and_num_return_sequence(),
                                                          get_multinomial_all_parameters(),
                                                        ],
                                                       prompts=[
                                                           "What is OpenVINO?",
                                                            "How are you?",
                                                            "Tell me something about Canada?",
                                                            "What is the time",
                                                            "Location is",
                                                       ],
                                                       ref_texts=[ 
                                                            ["\n\nOpenVINO is a high-level, middle-level, and customizable virtualization gateway. OpenVINO is a multi-stage virtual"],
                                                            ["   Are you not a freak and a liar?   I find it hard to believe that you're not, like, a very smart guy"],
                                                            ["\nI'm Canadian and I'm not sure what you mean by that.\nI'm not Canadian and I'm not sure what you mean by that"],
                                                            [
                                                                " in 30 minutes?\n78\nWhat is the second smallest value in -1/14, 2, 1.3, 0.3?\n",
                                                                " between 3:30AM and 4:40AM?\n8:30am—Sending orders\n9:30am—Sending orders\n",
                                                                " frame you are planning to take on?\nIm guessing. Im moving to Richmond next month so definitely not ready for winter yet",
                                                            ],
                                                            [
                                                                " important.\nI'm using both of those methods for now but if anything changes they'll be fine once we get more info about how their systems work",
                                                                " not in my opinion an issue at all since I am living with roommates that are close to where this game will land based off what they've heard",
                                                                " a bitch, especially when you have multiple devices running on each network...  That said: do some research before buying because its really hard (if very",
                                                                " the same as your first one and it's only 5 minutes away from me! Just go there :) or maybe just stay out until after midnight? lol",
                                                            ]
                                                       ])

@pytest.mark.parametrize("dynamic_split_fuse", [False, True])
@pytest.mark.precommit
def test_preemption_with_multinomial(tmp_path, dynamic_split_fuse):
    generation_configs = multinomial_params.generation_config
    for config in generation_configs:
        config.rng_seed = 0
    model_id : str = "facebook/opt-125m"
    model, hf_tokenizer = get_model_and_tokenizer(model_id, use_optimum=True)

    model_path : Path = tmp_path / model_id
    save_ov_model_from_optimum(model, hf_tokenizer, model_path)

    scheduler_config = get_scheduler_config({"num_kv_blocks": 30, "block_size": 32, "dynamic_split_fuse": dynamic_split_fuse, "max_num_batched_tokens": 256, "max_num_seqs": 256})
    generate_and_compare_with_reference_text(model_path, multinomial_params.prompts, multinomial_params.ref_texts, generation_configs, scheduler_config)
