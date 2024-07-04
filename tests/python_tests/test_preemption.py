# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from dataclasses import dataclass
from typing import List

from common import get_model_and_tokenizer, save_ov_model_from_optimum, generate_and_compare_with_reference_text, \
    DEFAULT_SCHEDULER_CONFIG, get_scheduler_config, run_test_pipeline, get_models_list, get_beam_search, get_greedy, \
    get_multinomial_all_parameters, get_multinomial_temperature_and_num_return_sequence, \
    get_multinomial_temperature_and_top_k, get_multinomial_temperature, get_multinomial_temperature_and_top_p
from test_sampling import RandomSamplingTestStruct

scheduler_params_list = [({"num_kv_blocks": 2, "block_size": 32, "dynamic_split_fuse": True, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_greedy()),
                         ({"num_kv_blocks": 2, "block_size": 32, "dynamic_split_fuse": False, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_greedy()),
                         ({"num_kv_blocks": 34, "block_size": 32, "dynamic_split_fuse": True, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_beam_search()),
                         ({"num_kv_blocks": 34, "block_size": 32, "dynamic_split_fuse": False, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_beam_search())]
@pytest.mark.parametrize("params", scheduler_params_list)
@pytest.mark.precommit
def test_preemption(tmp_path, params):
    run_test_pipeline(tmp_path, "facebook/opt-125m", params[0], params[1])

multinomial_params = RandomSamplingTestStruct(generation_config=[get_multinomial_temperature(),
                                                          get_multinomial_temperature_and_top_p(),
                                                          get_multinomial_temperature_and_top_k()],
                                                       prompts=["What is OpenVINO?",
                                                                "How are you?",
                                                                "Tell me something about Canada?",
                                                                ],
                                                       ref_texts=[ ["\n\nOpenVINO is a live platform that allows users to create and manage a new library for open source applications.\n\nOpenVINO is"],
                                                                   ["  You're getting much better results from doing this, than you are by not doing this.  I have a BH and I was so far"],
                                                                   ["\nI'm from Canada, and I'm from the US, so I'm not sure.\nI think you mean the Canadian version."]])



# todo: Anastasiia Pnevskaya: fix the test because it is hanging according max_new_tokens = std::numeric_limits<std::size_t>::max()
@pytest.mark.parametrize("dynamic_split_fuse", [True, False])
@pytest.mark.precommit
def test_preemption_with_multinomial(tmp_path, dynamic_split_fuse):
    generation_configs = multinomial_params.generation_config
    for config in generation_configs:
        config.rng_seed = 0
        config.max_new_tokens = 30
    model_id : str = "facebook/opt-125m"
    model, hf_tokenizer = get_model_and_tokenizer(model_id, use_optimum=True)

    model_path : Path = tmp_path / model_id
    save_ov_model_from_optimum(model, hf_tokenizer, model_path)

    scheduler_config = get_scheduler_config({"num_kv_blocks": 3, "block_size": 32, "dynamic_split_fuse": dynamic_split_fuse, "max_num_batched_tokens": 256, "max_num_seqs": 256})
    generate_and_compare_with_reference_text(model_path, multinomial_params.prompts, multinomial_params.ref_texts, generation_configs, scheduler_config)

multinomial_params_n_seq = RandomSamplingTestStruct(generation_config=[
        get_multinomial_temperature(),
        get_multinomial_temperature_and_num_return_sequence(),
        get_multinomial_all_parameters(),
    ],
    prompts=[
            "Artificial intelligence ",
            "What is the current",
            "Tell me something about UAE?",
            ],
    ref_texts=[
        [
            "\nI've seen this expression used too many times without making sense.\nAs an AI engineer, and as a scientist, we should all be looking"
        ],
        [
            ' significance of 3862?\n3829\nWhat is the greatest common divisor of 15 and 7763?\n9\nCalculate the',
            ' third derivative of 939*v**3*r**2 + 133*v**3*r**2 + v**3 - 77*',
            " climate in the future?  Do we have things to catch on fire, and if so does that mean we'll have a new climate before we have"
        ],
        [
            "\nIt's in the middle of nowhere if you haven’t seen one yet! It might be more convenient there than anywhere else 😊 we",
            '\nUAE is a country with some great culture that has been living under Islamic oppression for almost 60 years now (including 20 years before) so no',
            "\nI don't know anything.  I'm not sure what kind this sub wants though... but apparently they are pretty bad at taking selfies too..",
            '\nNope, just wanted to say how awesome and beautiful it was when my brother came back from an adventure trip across Asia - very much alive on'
        ],
    ])

@pytest.mark.skip(reason="should be fixed by support of n seqs in preemption")
@pytest.mark.parametrize("dynamic_split_fuse", [True, False])
@pytest.mark.precommit
def test_preemption_with_multinomial_n_seq(tmp_path, dynamic_split_fuse):
    generation_configs = multinomial_params_n_seq.generation_config
    for config in generation_configs:
        config.rng_seed = 0
    model_id : str = "facebook/opt-125m"
    model, hf_tokenizer = get_model_and_tokenizer(model_id, use_optimum=True)

    model_path : Path = tmp_path / model_id
    save_ov_model_from_optimum(model, hf_tokenizer, model_path)

    # needed kv_blocks - 16 (2 blocks per sequence (30 tokens to generated text + prompt (> 2 tokens)) * (1 + 3 + 4) seq )
    scheduler_config = get_scheduler_config({"num_kv_blocks": 8, "block_size": 32, "dynamic_split_fuse": dynamic_split_fuse, "max_num_batched_tokens": 256, "max_num_seqs": 256})
    generate_and_compare_with_reference_text(model_path, multinomial_params_n_seq.prompts, multinomial_params_n_seq.ref_texts, generation_configs, scheduler_config)