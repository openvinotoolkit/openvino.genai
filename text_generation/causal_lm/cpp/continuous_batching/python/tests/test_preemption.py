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
                                                          get_multinomial_temperature_and_top_k()
                                                        ],
                                                       prompts=[
                                                           "What is OpenVINO?",
                                                            "How are you?",
                                                            "Tell me something about Canada?",
                                                       ],
                                                       ref_texts=[ 
                                                            ["\n\nOpenVINO is a live platform that allows users to create and manage a new library for open source applications.\n\nOpenVINO is"],
                                                            ["  You're getting much better results from doing this, than you are by not doing this.  I have a BH and I was so far"],
                                                            ["\nI'm from Canada, and I'm from the US, so I'm not sure.\nI think you mean the Canadian version."]
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

    scheduler_config = get_scheduler_config({"num_kv_blocks": 3, "block_size": 32, "dynamic_split_fuse": dynamic_split_fuse, "max_num_batched_tokens": 256, "max_num_seqs": 256})
    generate_and_compare_with_reference_text(model_path, multinomial_params.prompts, multinomial_params.ref_texts, generation_configs, scheduler_config)

multinomial_params_n_seq = RandomSamplingTestStruct(generation_config=[
                                                          get_multinomial_temperature_and_num_return_sequence(),
                                                          get_multinomial_all_parameters(),
                                                    ],
                                                    prompts=[
                                                        "What is the time",
                                                        "Location is",
                                                    ],
                                                    ref_texts=[
                                                        [
                                                            " scale for a ditto?\nIt typically takes for it to be ready for ditto breeding.",
                                                            " limit for the break?\nSome time limits for me. Sometimes I wait for an hour and then it rains for 2 hours, sometimes I wait until",
                                                            " and how much of a problem with the frequency is your cells being overloaded?\nYes, it is intense. We all have troubles with our cells.",
                                                        ],
                                                        [
                                                            " correct.\nThank you!! It was my first time seeing this guy on Reddit from reddit (I'm new) lol he just happens by himself sometimes",
                                                            " pretty accurate with that!  Edit: You've got to make sure your phone has GPS coordinates where possible too since most people use their phones anyway haha",
                                                            " important, so if it's a different area of town then probably not as far away in person...but still have fun :) And when i did get",
                                                            " important and I can't find one for the US but somewhere around here they sell some cool cars like these :D What are those wheels called? Looks",
                                                        ]
                                                    ])

@pytest.mark.parametrize("dynamic_split_fuse", [False, True])
@pytest.mark.precommit
def test_preemption_with_multinomial_n_seq(tmp_path, dynamic_split_fuse):
    generation_configs = multinomial_params_n_seq.generation_config
    for config in generation_configs:
        config.rng_seed = 0
    model_id : str = "facebook/opt-125m"
    model, hf_tokenizer = get_model_and_tokenizer(model_id, use_optimum=True)

    model_path : Path = tmp_path / model_id
    save_ov_model_from_optimum(model, hf_tokenizer, model_path)

    scheduler_config = get_scheduler_config({"num_kv_blocks": 30, "block_size": 32, "dynamic_split_fuse": dynamic_split_fuse, "max_num_batched_tokens": 256, "max_num_seqs": 256})
    generate_and_compare_with_reference_text(model_path, multinomial_params_n_seq.prompts, multinomial_params_n_seq.ref_texts, generation_configs, scheduler_config)