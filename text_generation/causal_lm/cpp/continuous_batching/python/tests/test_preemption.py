# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from dataclasses import dataclass
from py_continuous_batching import GenerationConfig, GenerationResult
from typing import List

from common import get_model_and_tokenizer, save_ov_model_from_optimum, generate_and_compare_with_reference_text, \
    DEFAULT_SCHEDULER_CONFIG, get_scheduler_config, run_test_pipeline, get_models_list, get_beam_search, get_greedy, \
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


@pytest.mark.precommit
def test_out_of_memory(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        run_test_pipeline(tmp_path, "facebook/opt-125m", {"num_kv_blocks": 1})
    assert "Not enough memory for processing the requests." in str(excinfo.value)

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

@pytest.mark.parametrize("dynamic_split_fuse", [True, False])
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
