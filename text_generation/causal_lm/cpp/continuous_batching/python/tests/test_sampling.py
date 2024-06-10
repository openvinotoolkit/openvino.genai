# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from dataclasses import dataclass
from pathlib import Path
from py_continuous_batching import GenerationConfig
from typing import List

from common import run_test_pipeline, get_models_list, get_model_and_tokenizer, save_ov_model_from_optimum, \
    generate_and_compare_with_reference_text, get_greedy, get_beam_search, get_multinomial_temperature, \
    get_multinomial_temperature_and_top_k, get_multinomial_temperature_and_top_p, \
    get_multinomial_temperature_top_p_and_top_k, DEFAULT_SCHEDULER_CONFIG, get_greedy_with_repetition_penalty, \
    get_multinomial_all_parameters, get_multinomial_temperature_and_num_return_sequence, \
    generate_and_compare_with_hf, get_multinomial_temperature_and_repetition_penalty, get_scheduler_config


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
    generate_and_compare_with_hf(model_id, prompts, generation_configs, scheduler_config, tmp_path)


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
    generate_and_compare_with_hf(model_id, prompts, generation_configs, scheduler_config, tmp_path)

@pytest.mark.precommit
@pytest.mark.parametrize("generation_config", [get_greedy(), get_beam_search(), get_greedy_with_repetition_penalty()],
        ids=["greedy", "beam", "greedy_with_repetition_penalty"])
def test_individual_generation_configs_deterministic(tmp_path, generation_config):
    prompts = [
            "What is OpenVINO?",
            ]
    generation_configs = [generation_config]
    model_id : str = "facebook/opt-125m"
    generate_and_compare_with_hf(model_id, prompts, generation_configs, DEFAULT_SCHEDULER_CONFIG, tmp_path)


@dataclass
class RandomSamplingTestStruct:
    generation_config: GenerationConfig
    prompts: List[str]
    ref_texts: List[List[str]]

RANDOM_SAMPLING_TEST_CASES = [
    RandomSamplingTestStruct(generation_config=get_multinomial_temperature(),
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
    RandomSamplingTestStruct(generation_config=get_multinomial_temperature_and_repetition_penalty(),
                             prompts=["What is OpenVINO?"],
                             ref_texts=[ ["\nOpen Vino's are a new and improved way to find cheap, fast-investment frozen vegetables that have no waste or calories. They're"] ]),
    RandomSamplingTestStruct(generation_config=get_multinomial_temperature_and_num_return_sequence(),
                             prompts=["What is location of"],
                             ref_texts=[
                                [
                                    ' your instruments?  Are they in an armpit?  Is it warm?  Are your instruments clear?  Are there any cuts and scratches',
                                    ' map and where does the game player base base?    I tend to like to do all draws on a specific spot (sometimes wide area,',
                                    ' them?\nJust the Mario Maker App, the location is they'
                                ]
                             ]),
    RandomSamplingTestStruct(generation_config=get_multinomial_all_parameters(),
                             prompts=["Tell me something about UAE"],
                             ref_texts=[
                                [
                                    " and how it's not like we're all in the same boat right now lol (or even close) 😂😁! Just curious :) If",
                                    "?  You are my country... so what does our military do here?? What am i missing out on?? And why don't u tell us?",
                                    '?\nThe U.S government has been doing quite well with foreign-made aircraft for many years under US administration....and they have very good reasons',
                                    '? I think that is a bit of an anomaly, but you might want to ask yourself this question: Where can some young people from Dubai or Bahrain'
                                ]
                             ]),
]


@pytest.mark.precommit
@pytest.mark.parametrize("test_struct", RANDOM_SAMPLING_TEST_CASES,
        ids=["multinomial_temperature",
             "multinomial_temperature_and_top_p",
             "multinomial_temperature_and_top_k",
             "multinomial_temperature_top_p_and_top_k",
             "multinomial_temperature_and_repetition_penalty",
             "multinomial_temperature_and_num_return_sequence",
             "multinomial_all_parameters",
            "multinomial_temperature_and_presence_penalty",
            "multinomial_temperature_and_frequence_penalty"])
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
