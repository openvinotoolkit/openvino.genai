# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import pytest

from common import run_test_pipeline, get_models_list, get_model_and_tokenizer, save_ov_model_from_optimum, generate_and_compare_with_reference_text, get_greedy, get_beam_search, get_multinomial_temperature, get_multinomial_temperature_and_top_k, get_multinomial_temperature_and_top_p, get_multinomial_temperature_top_p_and_top_k, DEFAULT_SCHEDULER_CONFIG
from dataclasses import dataclass
from py_continuous_batching import GenerationConfig, GenerationResult
from pathlib import Path
from typing import List



@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_models_list(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "precommit")))
def test_sampling_precommit(tmp_path, model_id):
    run_test_pipeline(tmp_path, model_id)


@pytest.mark.nightly
@pytest.mark.parametrize("model_id", get_models_list(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "nightly")))
def test_sampling_nightly(tmp_path, model_id):
    run_test_pipeline(tmp_path, model_id)

@pytest.mark.real_models
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

@pytest.mark.parametrize("generation_config", [get_greedy(), get_beam_search()],
        ids=["greedy", "beam"])
def test_individual_generation_configs_deterministic(tmp_path, generation_config):
    prompts = [
            "What is OpenVINO?",
            ]
    generation_configs = [generation_config]
    model_id : str = "facebook/opt-125m"
    _generate_and_compare_with_hf(model_id, prompts, generation_configs, DEFAULT_SCHEDULER_CONFIG, tmp_path)


@dataclass
class RandomSamplingTestStruct:
    generation_config: GenerationConfig
    prompts: List[str]
    ref_texts: List[List[str]]

RANDOM_SAMPLING_TEST_CASES = [RandomSamplingTestStruct(generation_config=get_multinomial_temperature(),
                                                       prompts=["What is OpenVINO?"],
                                                       ref_texts=[ ["\n\nOpenVINO is a software development platform developed by OpenVINO, a set of technology companies and startups that enables developers to use the most"] ]),
                              RandomSamplingTestStruct(generation_config=get_multinomial_temperature_and_top_p(),
                                                       prompts=["What is OpenVINO?"],
                                                       ref_texts=[ ["\nOpenVINO is an online application that allows users to create, test, and analyze their own software using a collection of software packages. The application"] ]),
                              RandomSamplingTestStruct(generation_config=get_multinomial_temperature_and_top_k(),
                                                       prompts=["What is OpenVINO?"],
                                                       ref_texts=[ ["\n\nOpenVINO is a software that allows users to create a virtual machine with the ability to create a virtual machine in a virtual environment. Open"] ]),
                              RandomSamplingTestStruct(generation_config=get_multinomial_temperature_top_p_and_top_k(),
                                                       prompts=["What is OpenVINO?"],
                                                       ref_texts=[ ["\nOpenVINO is an open source software that allows developers to create, manage, and distribute software. It is an open source project that allows developers"] ]),
                              ]


@pytest.mark.precommit
@pytest.mark.parametrize("test_struct", RANDOM_SAMPLING_TEST_CASES,
        ids=["multinomial_temperature", "multinomial_temperature_and_top_p", "multinomial_temperature_and_top_k", "multinomial_temperature_top_p_and_top_k"])
def test_individual_generation_configs_random(tmp_path, test_struct: RandomSamplingTestStruct):
    generation_config = test_struct.generation_config

    prompts = test_struct.prompts
    generation_config.rng_seed = 0
    generation_configs = [generation_config]
    model_id : str = "facebook/opt-125m"
    model, hf_tokenizer = get_model_and_tokenizer(model_id, use_optimum=True)

    model_path : Path = tmp_path / model_id
    save_ov_model_from_optimum(model, hf_tokenizer, model_path)

    generate_and_compare_with_reference_text(model_path, prompts, test_struct.ref_texts, generation_configs, DEFAULT_SCHEDULER_CONFIG)
