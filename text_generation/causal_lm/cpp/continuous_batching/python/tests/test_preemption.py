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
                                                            "What is the time of",
                                                            "Location is",
                                                       ],
                                                       ref_texts=[ 
                                                            ["\n\nOpenVINO, a VINOS server is a Linux-based virtual bootstation, helping restore virtual machines from boot failure and restore system"],
                                                            ["    I have been aod' for a year now and haven't heard anything about you.    Is it really that easy"],
                                                            ["\nI'm Canadian and I'm not a fan of the Canadian flag.\nI'm Canadian and I'm not a Canadian.\nI'm Canadian"],
                                                            [
                                                                " year it was colder than that for the n4?\nThe time of the year is the same as the time of the year the original OG N",
                                                                " day you play?\nWhen I'm home.  For the most part I'm playing for about 5 hours.  I might change it later tonight",
                                                                " day the bugs are present? Do those bug me? Do they fly away?\nThey're not! Well, they're usually left alone in the"
                                                            ],
                                                            [
                                                                " a problem with your network of friends and family who may have been affected by this incident? If so please contact our Customer Service team within 24 hours immediately",
                                                                " important, but it can also be more difficult to get an accurate reading on what’s going through their minds at night compared from day-to",
                                                                " the same as everything else in life except for being closer than you think (in most ways). The only difference between here or there isn't how close",
                                                                " not required.\nIt\'s just that I\'m really looking forward doing my part towards supporting those people without having any kinda \"support\". It would"
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
