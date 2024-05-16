# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import pytest

from common import (
    run_test_pipeline,
    run_hugging_face,
    run_continuous_batching,
    get_models_list,
    get_greedy,
    get_beam_search,
    get_scheduler_config,
    compare_results
)

# tested models:
# - facebook/opt-125m
# - meta-llama/Llama-2-7b-chat-hf
# - mistralai/Mistral-7B-Instruct-v0.2

@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_models_list(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "precommit")))
def test_sampling_precommit(tmp_path, model_id):
    run_test_pipeline(tmp_path, model_id)


@pytest.mark.nightly
@pytest.mark.parametrize("model_id", get_models_list(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "nightly")))
def test_sampling_nightly(tmp_path, model_id):
    run_test_pipeline(tmp_path, model_id)


@pytest.mark.parametrize("model_id", get_models_list(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "real_models")))
def test_real_models(tmp_path, model_id):
    run_test_pipeline(tmp_path, model_id)


@pytest.mark.precommit
def test_eos_beam_search(tmp_path):
    '''
    Current test checks that in case of beam search, some generation results
    explicitly have EOS token at the end, which is aligned with HF

    Example of current output:
    { -1.23264,  that I don't know about.
    I don't know what you're talking about, but I'm pretty sure it's a Canadian thing.</s> }
    '''
    model_id = "facebook/opt-125m"
    prompts = ["Tell me something about Canada"]
    generation_configs = [get_beam_search()]
    scheduler_config = get_scheduler_config()

    (hf_results, model_path) = run_hugging_face(model_id=model_id, prompts=prompts,
                                                generation_configs=generation_configs, tmp_path=tmp_path,
                                                use_optimum=True)
    ov_results: List[GenerationResult] = run_continuous_batching(model_path=model_path, scheduler_config=scheduler_config,
                                                                 prompts=prompts, generation_configs=generation_configs)

    assert len(prompts) == len(hf_results)
    assert len(prompts) == len(ov_results)

    for prompt, hf_result, ov_result, generation_config in zip(prompts, hf_results, ov_results, generation_configs):
        print(f"Prompt = {prompt}\nHF result = {hf_result}\nOV result = {ov_result}")
        compare_results(hf_result, ov_result, generation_config)


@pytest.mark.precommit
def test_eos_greedy(tmp_path):
    '''
    Current test checks that in case of gready, some generation results
    explicitly have EOS token at the end, which is aligned with HF:

    Example of current output:
    {  a software program</s> }
    '''
    model_id = "bigscience/bloomz-560m"
    prompts = ["What is OpenVINO?"]
    generation_configs = [get_greedy()]
    scheduler_config = get_scheduler_config()

    (hf_results, model_path) = run_hugging_face(model_id=model_id, prompts=prompts,
                                                generation_configs=generation_configs, tmp_path=tmp_path,
                                                use_optimum=True)
    ov_results: List[GenerationResult] = run_continuous_batching(model_path=model_path, scheduler_config=scheduler_config,
                                                                 prompts=prompts, generation_configs=generation_configs)

    assert len(prompts) == len(hf_results)
    assert len(prompts) == len(ov_results)

    for prompt, hf_result, ov_result, generation_config in zip(prompts, hf_results, ov_results, generation_configs):
        print(f"Prompt = {prompt}\nHF result = {hf_result}\nOV result = {ov_result}")
        compare_results(hf_result, ov_result, generation_config)
