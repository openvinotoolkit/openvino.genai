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
                                                            "What do you know about OpenVINO?",
                                                            "How are you?",
                                                            "Tell me something about Canada?",
                                                            "What is the time of",
                                                            "Location is",
                                                       ],
                                                       ref_texts=[ 
                                                            ["\nIt's a popular tool that scales up to a new and improved version of the GPU drivers before installing the full version and doesn't crash or make"],
                                                            ["   Are you not a person with PTSD?\nI am a person with PTSD but no I am not a person with PTSD. I have had"],
                                                            ["\nI'm Canadian and I'm a bit of a fan of the Canadian culture.\nI'm a bit of a fan of the Canadian culture."],
                                                            [
                                                                ' day you plan to download this on?\nOne day in the morning.\nOk. When does it arrive on?\nIt arrived in the mail',
                                                                ' the week that you are eligible for the lottery?\n\nIf you die on Fridays, Monday and Tuesday (28 June, 27 July and 31 July',
                                                                ' day when you wake up and struggle to understand what happened?\nWhen I wake up I am silently trying to comprehend what I saw.\nIt can'
                                                            ],
                                                            [
                                                                ' not enough\nI agree!  Just as far as location goes...it really depends what people want their games for (and who owns / modd',
                                                                " important, but you're looking at a long distance connection between both companies if your company doesn't offer anything close by from here yet :/\nMy",
                                                                " important. I'm in Australia and my friend lives on another continent so we just have to find where she can stay during that time period instead of worrying",
                                                                ' wrong - the website states it\'s "in California". Where are they located? It looks like there was an error with them when ordering online through D'
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
