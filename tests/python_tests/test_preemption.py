# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest

from openvino_genai import GenerationConfig
from common import get_model_and_tokenizer, save_ov_model_from_optimum, generate_and_compare_with_reference_text, \
    DEFAULT_SCHEDULER_CONFIG, get_scheduler_config, run_test_pipeline, get_models_list, get_beam_search, get_greedy, \
    get_multinomial_all_parameters, get_multinomial_temperature_and_num_return_sequence, \
    get_multinomial_temperature_and_top_k, get_multinomial_temperature, get_multinomial_temperature_and_top_p
from test_sampling import RandomSamplingTestStruct

def get_greedy_seq_len_300() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 3
    generation_config.max_new_tokens = 300
    return generation_config

def get_beam_search_seq_len_300() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_beam_groups = 3
    generation_config.num_beams = 6
    generation_config.max_new_tokens = 300
    generation_config.num_return_sequences = generation_config.num_beams
    return generation_config

scheduler_params_list = [({"num_kv_blocks": 2, "block_size": 32, "dynamic_split_fuse": True, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_greedy()),
                         ({"num_kv_blocks": 2, "block_size": 32, "dynamic_split_fuse": False, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_greedy()),
                         ({"num_kv_blocks": 10, "block_size": 32, "dynamic_split_fuse": True}, get_greedy_seq_len_300()),
                         ({"num_kv_blocks": 10, "block_size": 32, "dynamic_split_fuse": False}, get_greedy_seq_len_300()),
                         ({"num_kv_blocks": 34, "block_size": 32, "dynamic_split_fuse": True, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_beam_search()),
                         ({"num_kv_blocks": 34, "block_size": 32, "dynamic_split_fuse": False, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_beam_search()),
                         ({"num_kv_blocks": 100, "block_size": 32, "dynamic_split_fuse": True}, get_beam_search_seq_len_300()),
                         ({"num_kv_blocks": 100, "block_size": 32, "dynamic_split_fuse": False}, get_beam_search_seq_len_300())]
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
@pytest.mark.xfail(raises=AssertionError, reason="assert ref_text == ov_text fails in CI.", condition=sys.platform in ["win32", "darwin"], strict=True)
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
            "\nI've seen this expression used too many times without making sense.\nAs an AI engineer, and as a scientist, we should make everything easier"
        ],
        [
            ' significance of 3862?\n3829\nWhat is the greatest common divisor of 15 and 7763?\n9\nCalculate the',
            ' third derivative of 939*v**3*r**2 + 133*v**3*r**2 + v**3 - 16*',
            " climate in the future?  Do we have things to catch on fire, and if so does that mean we'll have a new climate change or is"
        ],
        [
            "\nIt's in the middle of nowhere if you haven’t seen one yet! It might be more convenient there than anywhere else.. maybe take",
            '\nUAE is a country with some great culture that has been living under Islamic oppression for almost 60 years now (including 20 years as part of Arab',
            '\nNope, just wanted to say how awesome and beautiful it was when my brother came back from an adventure trip across Asia - our 2nd year',
            '\nI don\'t know anything.  I\'m not sure what kind this sub wants though... but apparently they are pretty bad at making videos/photos'
        ],
    ])

@pytest.mark.parametrize("dynamic_split_fuse", [True, False])
@pytest.mark.precommit
@pytest.mark.xfail(reason="assert ref_text == ov_text fails", condition=sys.platform in ["win32", "darwin"])
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