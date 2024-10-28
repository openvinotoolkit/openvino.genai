# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from openvino_genai import GenerationConfig
from common import get_model_and_tokenizer, save_ov_model_from_optimum, generate_and_compare_with_reference_text, \
    get_scheduler_config, run_test_pipeline, get_beam_search, get_greedy, \
    get_multinomial_all_parameters, get_multinomial_temperature_and_num_return_sequence, \
    get_multinomial_temperature_and_top_k, get_multinomial_temperature, get_multinomial_temperature_and_top_p
from test_sampling import RandomSamplingTestStruct, get_current_platform_ref_texts


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


multinomial_params = RandomSamplingTestStruct(
    generation_config=[
        get_multinomial_temperature(),
        get_multinomial_temperature_and_top_p(),
        get_multinomial_temperature_and_top_k(),
    ],
    prompts=[
        "What is OpenVINO?",
        "How are you?",
        "Tell me something about Canada?",
    ],
    ref_texts=get_current_platform_ref_texts({
        "linux": [
            [
                "\n\nOpenVINO is a live platform that allows users to create and manage a new library for open source applications.\n\nOpenVINO is"
            ],
            [
                "  You're getting much better results from doing this, than you are by not doing this.  I have a BH and I was so far"
            ],
            [
                "\nI'm from Canada, and I'm from the US, so I'm not sure.\nI think you mean the Canadian version."
            ],
        ],
        "win32": [
            [
                "\n\nOpenVINO is a live platform that allows users to create and manage a new library of applications on the Virtuoso server, which can"
            ],
            [
                "  You're getting much better results from doing this, than you are by not doing this.  If you are truly trying to do something good,"
            ],
            [
                "\nI'm from Canada, and I'm from the US, so I'm not sure what you're talking about.\nI'm Canadian and I"
            ],
        ],
    }),
)


# todo: Anastasiia Pnevskaya: fix the test because it is hanging according max_new_tokens = std::numeric_limits<std::size_t>::max()
@pytest.mark.parametrize("dynamic_split_fuse", [True, False])
@pytest.mark.precommit
@pytest.mark.skip(reason="Random sampling results are non deterministic due to: discrete_distribution impl depends on platform, model inference results may depend on CPU. Test passes on CI but fails locally.")
def test_preemption_with_multinomial(tmp_path, dynamic_split_fuse):
    generation_configs = multinomial_params.generation_config
    for config in generation_configs:
        config.rng_seed = 0
        config.max_new_tokens = 30
    model_id : str = "facebook/opt-125m"
    model, hf_tokenizer = get_model_and_tokenizer(model_id, use_optimum=True)

    models_path : Path = tmp_path / model_id
    save_ov_model_from_optimum(model, hf_tokenizer, models_path)

    scheduler_config = get_scheduler_config({"num_kv_blocks": 3, "block_size": 32, "dynamic_split_fuse": dynamic_split_fuse, "max_num_batched_tokens": 256, "max_num_seqs": 256})
    generate_and_compare_with_reference_text(models_path, multinomial_params.prompts, multinomial_params.ref_texts, generation_configs, scheduler_config)


multinomial_params_n_seq = RandomSamplingTestStruct(
    generation_config=[
        get_multinomial_temperature(),
        get_multinomial_temperature_and_num_return_sequence(),
        get_multinomial_all_parameters(),
    ],
    prompts=[
        "Artificial intelligence ",
        "What is the current",
        "Tell me something about UAE?",
    ],
    ref_texts=get_current_platform_ref_texts({
        "linux": [
            [
                "\nI've seen this expression used too many times without making sense.\nAs an AI engineer, and as a scientist, we should make everything easier"
            ],
            [
                " position of the Z-shaped groove?\n0.41\nWhat is the current position of the Z-shaped groove?\n0.11\n",
                " status of all of this? I can't stop thinking about it.\nIt's been a while since I've seen it. I found it a",
                " status of your blog? Do you accept feedback?\nYes, I’m happy to accept feedback at this time (I’m a"
            ],
            [
                "\nIt's in the middle of nowhere if you haven’t seen one yet! It might be more convenient there than anywhere else.. maybe take",
                "\nUAE is a country with some great culture that has been living under Islamic oppression for almost 60 years now (including 20 years as part of Arab",
                "\nNope, just wanted to say how awesome and beautiful it was when my brother came back from an adventure trip across Asia - our 2nd year",
                "\nI don't know anything.  I'm not sure what kind this sub wants though... but apparently they are pretty bad at making videos/photos",
            ],
        ],
        "win32": [
            [
                "\nI've had a friend with the capacity to test this in his own words.\nThe big problem with real-world results is the economics of"
            ],
            [
                " position of the patent application number of the present invention?\n\nIn the present invention, the present invention relates to an improved method for manufacturing a semic",
                " status of your town? How many houses do you have?\nThere are about three houses in our town. The closest place to us is about 25",
                " status of all the other passengers?\nWe're the only ones left, so no...\nI don't think they'll really leave.\nThey"
            ],
            [
                "\nI don't have any knowledge on them. We are based out near Dubai so hopefully they will take care of us soon enough :) thanks though :",
                "\nUAE is not one of the richest countries in Asia but definitely among those most corrupt nations because this corruption (and its own endemic practices) still",
                "\nNope, I'm just going through my first semester there right now and it was nice to see some people who were doing well haha - we",
                "\nIt's a country where your parents can never give you anything at all!  It also has an extremely low education system for many years... You",
            ],
        ],
        "darwin": [
            [
                "\nI've had a friend with the capacity to test this in his own words.\nThe big problem with real-world results is the rigidity"
            ],
            [
               " position of the patent application number of the present invention?\n\nIn the present invention, the present invention relates to an improved method for manufacturing a semic",
               " status of your town? How many houses do you have?\nThere are about three houses in our town. The closest place to us is about 25",
               " status of all the other passengers?\nWe're the only ones left, so no...\nI don't think they'll really leave.\nThey"
            ],
            [
                "\nI don't have any knowledge on them. We are based out near Dubai so hopefully they will take care of us soon enough :) thanks though :",
                "\nUAE is not one of the richest countries in Asia but definitely among those most corrupt nations because this corruption (and its own endemic practices) still",
                "\nNope, I'm just going through my first semester there right now and it was nice to see some people who were doing well haha - we",
                "\nIt's a country where your parents can never give you anything at all!  It also has an extremely low education system for many years... You",
            ],
        ],
    }),
)


@pytest.mark.parametrize("dynamic_split_fuse", [True, False])
@pytest.mark.precommit
@pytest.mark.skip(reason="Random sampling results are non deterministic due to: discrete_distribution impl depends on platform, model inference results may depend on CPU. Test passes on CI but fails locally.")
def test_preemption_with_multinomial_n_seq(tmp_path, dynamic_split_fuse):
    generation_configs = multinomial_params_n_seq.generation_config
    for config in generation_configs:
        config.rng_seed = 0
    model_id : str = "facebook/opt-125m"
    model, hf_tokenizer = get_model_and_tokenizer(model_id, use_optimum=True)

    models_path : Path = tmp_path / model_id
    save_ov_model_from_optimum(model, hf_tokenizer, models_path)

    # needed kv_blocks - 16 (2 blocks per sequence (30 tokens to generated text + prompt (> 2 tokens)) * (1 + 3 + 4) seq )
    scheduler_config = get_scheduler_config({"num_kv_blocks": 8, "block_size": 32, "dynamic_split_fuse": dynamic_split_fuse, "max_num_batched_tokens": 256, "max_num_seqs": 256})
    generate_and_compare_with_reference_text(models_path, multinomial_params_n_seq.prompts, multinomial_params_n_seq.ref_texts, generation_configs, scheduler_config)