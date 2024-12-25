# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import pytest
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from openvino_genai import ContinuousBatchingPipeline, GenerationConfig, Tokenizer
from typing import List, TypedDict

from common import get_hugging_face_model_and_tokenizer, save_ov_model_from_optimum, \
    get_greedy, get_beam_search, get_multinomial_temperature, \
    get_greedy_with_penalties, get_multinomial_temperature, \
    get_multinomial_temperature_and_top_k, get_multinomial_temperature_and_top_p, \
    get_multinomial_temperature_top_p_and_top_k, DEFAULT_SCHEDULER_CONFIG, get_greedy_with_repetition_penalty, \
    get_multinomial_all_parameters, get_multinomial_temperature_and_num_return_sequence, \
    get_greedy, get_greedy_with_min_and_max_tokens, \
    get_greedy_with_single_stop_string, get_greedy_with_multiple_stop_strings, get_greedy_with_multiple_stop_strings_no_match, \
    get_beam_search, get_beam_search_min_and_max_tokens, get_beam_search_with_single_stop_string, \
    get_beam_search_with_multiple_stop_strings, get_beam_search_with_multiple_stop_strings_no_match, get_multinomial_max_and_min_token, \
    get_multinomial_temperature_and_frequence_penalty, get_multinomial_temperature_and_presence_penalty, \
    get_greedy_stop_strings_exclude_from_output, get_greedy_stop_strings_include_to_output, \
    get_greedy_n_stop_strings_exclude_from_output, get_greedy_n_stop_strings_include_to_output, \
    generate_and_compare_with_hf, get_multinomial_temperature_and_repetition_penalty, get_scheduler_config, \
    run_continuous_batching


# TODO: currently, this test drops EOS token as both HF and OV use `skip_special_tokens=True`, which should be disabled for samlpling tests
@pytest.mark.precommit
def test_beam_search_has_eos_token_at_end(tmp_path):
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


# TODO: currently, this test drops EOS token as both HF and OV use `skip_special_tokens=True`, which should be disabled for samlpling tests
@pytest.mark.precommit
def test_greedy_has_eos_token_at_end(tmp_path):
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


# TODO: consider removing all these functions with generation configs and use Dict with properties, which can be converted to generation config
@pytest.mark.precommit
@pytest.mark.parametrize("generation_config",
                         [get_greedy(), get_greedy_with_min_and_max_tokens(), get_greedy_with_repetition_penalty(), get_greedy_with_penalties(), get_greedy_with_single_stop_string(),
                          get_greedy_with_multiple_stop_strings(), get_greedy_with_multiple_stop_strings_no_match(),
                          get_beam_search(), get_beam_search_min_and_max_tokens(), get_beam_search_with_multiple_stop_strings_no_match(),
                          get_greedy_stop_strings_exclude_from_output(), get_greedy_stop_strings_include_to_output(),
                          get_greedy_n_stop_strings_exclude_from_output(), get_greedy_n_stop_strings_include_to_output()],
                         ids=["greedy", "greedy_with_min_and_max_tokens", "greedy_with_repetition_penalty", "greedy_with_penalties", "greedy_with_single_stop_string",
                              "greedy_with_multiple_stop_strings", "greedy_with_multiple_stop_strings_no_match", "beam_search", "beam_search_min_and_max_tokens",
                              "beam_search_with_multiple_stop_strings_no_match", "greedy_stop_strings_exclude_from_output", "greedy_stop_strings_include_to_output",
                              "greedy_n_stop_strings_exclude_from_output", "greedy_n_stop_strings_include_to_output"])
def test_sampling_against_optimum(tmp_path, generation_config):
    prompts = [ "What is OpenVINO?" ]
    generation_configs = [generation_config]
    model_id : str = "facebook/opt-125m"
    generate_and_compare_with_hf(model_id, prompts, generation_configs, DEFAULT_SCHEDULER_CONFIG, tmp_path)


@pytest.mark.precommit
@pytest.mark.xfail(
    raises=AssertionError,
    reason="Stop strings do not seem to work as expected with beam search in HF, so comparison will fail. If it changes, these cases shall be merged to the test above.",
    strict=True,
)
@pytest.mark.parametrize("generation_config", [get_beam_search_with_single_stop_string(), get_beam_search_with_multiple_stop_strings()],
                         ids=["beam_search_with_single_stop_string", "beam_search_with_multiple_stop_strings"])
def test_beam_search_with_stop_string(tmp_path, generation_config):
    prompts = [ "What is OpenVINO?" ]
    generation_configs = [generation_config]
    model_id : str = "facebook/opt-125m"
    generate_and_compare_with_hf(model_id, prompts, generation_configs, DEFAULT_SCHEDULER_CONFIG, tmp_path)


# TODO: remove platform specific reference texts once CVS-159912 is done and use comparison with HF
# and merge this tests with 'test_sampling_against_optimum' by extending a list of generation configs

class PlatformsRefTexts(TypedDict, total=False):
    linux: List[List[str]]
    win32: List[List[str]]
    darwin: List[List[str]]


def get_current_platform_ref_texts(ref_texts: PlatformsRefTexts) -> List[List[str]]:
    # mac and win often have identical results
    # to avoid duplication, use win32 ref_text if no mac ref_texts were found
    if sys.platform == "darwin":
        result = ref_texts.get("darwin") or ref_texts.get("win32")
    else:
        result = ref_texts.get(sys.platform)
    if not result:
        raise RuntimeError("No ref_texts were provided")
    return result


@dataclass
class RandomSamplingTestStruct:
    generation_config: GenerationConfig
    prompts: List[str]
    ref_texts: List[List[str]]


RANDOM_SAMPLING_TEST_CASES = [
    RandomSamplingTestStruct(
        generation_config=get_multinomial_temperature(),
        prompts=["What is OpenVINO?"],
        ref_texts=[
            [
                "\n\nOpenVINO is a software development platform developed by OpenVINO, a set of technology companies and startups that enables developers to use the most"
            ]
        ],
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_temperature_and_top_p(),
        prompts=["What is OpenVINO?"],
        ref_texts=get_current_platform_ref_texts({
            "linux": [
                [
                    "\nOpenVINO is an online application that allows users to create, test, and analyze their own software using a collection of software packages. The application"
                ]
            ],
            "win32": [
                [
                    "\n\nOpenVINO is a software development platform designed to allow developers to develop and commercialize the most important software products on the web. OpenV"
                ]
            ],
        })
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_temperature_and_top_k(),
        prompts=["What is OpenVINO?"],
        ref_texts=[
            [
                "\n\nOpenVINO is a software that allows users to create a virtual machine with the ability to create a virtual machine in a virtual environment. Open"
            ]
        ],
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_temperature_top_p_and_top_k(),
        prompts=["What is OpenVINO?"],
        ref_texts=get_current_platform_ref_texts({
            "linux": [
                [
                    "\nOpenVINO is an open source software that allows developers to create, manage, and distribute software. It is an open source project that allows developers"
                ]
            ],
            "win32": [
                [
                    "\n\nOpenVINO is a software that allows users to create a virtual machine with the ability to create a virtual machine in a virtual environment. Open"
                ]
            ],
        }),
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_temperature_and_repetition_penalty(),
        prompts=["What is OpenVINO?"],
        ref_texts=[
            [
                "\nOpen Vino's are a new and improved way to find cheap, fast-investment frozen vegetables that have no waste or calories. They're"
            ]
        ],
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_temperature_and_num_return_sequence(),
        prompts=["What is location of"],
        ref_texts=[
            [
                " the exact same image?\nI've tried multiple times to find it, but I'm still not sure. I am sure it's the exact same",
                " your new house?\nAnywhere that has a GPS. It will be up to you.",
                " your cat?  He is more likely to be on the floor with him.\nTalduck"
            ]
        ],
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_all_parameters(),
        prompts=["Tell me something about UAE"],
        ref_texts=get_current_platform_ref_texts({
            "linux": [
                [
                    " and how it's not like we're all in the same boat right now lol (or even close) 😂😁! Just curious :) If",
                    "?  You are my country... so what does our military do here?? What am i missing out on?? And why don't u tell us?",
                    "?\nThe U.S government has been doing quite well with foreign-made aircraft for many years under US administration....and they have very good reasons",
                    "? I think that is a bit of an anomaly, but you might want to ask yourself this question: Where can some young people from Dubai or Bahrain",
                ]
            ],
            "win32": [
                [
                    "? I think that is a bit of an anomaly, especially since there aren't many Americans living here (like us). What makes you say they've",
                    "?  You are my country... so what does our future have to do with your problems?? \U0001f609\U0001f608\U0001f495 \U0001f5a4\ufffd",
                    "?\nThe U.S government has been doing quite well for decades now when compared strictly directly or indirectly as regards security issues.. They even made some",
                    " and how it's not like we're all in the same boat either! We had such fun meeting each other at different times this past summer :) It",
                ]
            ],
        }),
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_temperature_and_presence_penalty(),
        prompts=["What is OpenVINO?"],
        ref_texts=[
            [
                "\n\nOpenVINO is a software development platform developed by OpenVINO, Inc., which uses a RESTful API for server-side web applications"
            ]
        ],
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_temperature_and_frequence_penalty(),
        prompts=["What is OpenVINO?"],
        ref_texts=[
            [
                "\n\nOpenVINO is a software development platform developed by OpenVINO, Inc., which offers the Linux-based platform. OpenVINO's"
            ]
        ],
    ),
    RandomSamplingTestStruct(
        generation_config=get_multinomial_max_and_min_token(),
        prompts=["What is OpenVINO?"],
        ref_texts=get_current_platform_ref_texts({
            "linux": [
                [
                    "\nOpenVINO is a Linux distro. It's not as simple as using the Linux distro itself. OpenVINO is essentially a dist",
                    "\nOpenVINO is an open-source open-source software that allows anyone to work with a virtual machine, from a smartphone to an iPhone,",
                    "\n\nOpenVINO is a social networking tool. OpenVINO is a free virtualization service that works at scale. The tool provides the ability",
                ]
            ],
            "win32": [
                [
                    "\nOpenVINO is the latest addition to the OpenVINO series of platforms. OpenVINO is an open source software development framework for all platforms",
                    "\nOpenVINO is a browser-based virtual assistant that enables developers and developers to quickly communicate with their own virtual machines. Using this virtual assistant,",
                    "\n\nOpenVINO is a program designed to help you find the best open source open source software. The program, which is a lightweight package and",
                ]
            ],
        }),
    ),
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
             "multinomial_temperature_and_frequence_penalty",
             "multinomial_max_and_min_token"])
def test_multinomial_sampling_against_reference(tmp_path, test_struct: RandomSamplingTestStruct):
    generation_config = test_struct.generation_config

    prompts = test_struct.prompts
    generation_config.rng_seed = 0
    generation_configs = [generation_config]
    model_id : str = "facebook/opt-125m"
    model, hf_tokenizer = get_hugging_face_model_and_tokenizer(model_id, use_optimum=True)

    models_path : Path = tmp_path / model_id
    save_ov_model_from_optimum(model, hf_tokenizer, models_path)

    # run multinomial without comparison with reference
    _ = run_continuous_batching(models_path, DEFAULT_SCHEDULER_CONFIG, prompts, generation_configs)

    # Reference comparison is not performed as sampling results are non-deterministic.
    # Discrete_distribution impl depends on platform, model inference results may depend on CPU.


@pytest.mark.precommit
@pytest.mark.parametrize("get_generation_config", [get_greedy, get_beam_search, get_multinomial_all_parameters],
                         ids=["greedy", "beam_search", "multinomial_all_parameters"])
@pytest.mark.parametrize("max_num_batched_tokens", [2, 4, 256])
def test_echo_prompt_phase_only(tmp_path, get_generation_config, max_num_batched_tokens):
    generation_config = get_generation_config()
    generation_config.max_new_tokens = 0
    generation_config.echo = True

    scheduler_config = get_scheduler_config()
    scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    generation_configs = [generation_config]
    model_id : str = "facebook/opt-125m"
    opt_model, hf_tokenizer = get_hugging_face_model_and_tokenizer(model_id, use_optimum=True)

    model_path : Path = tmp_path / model_id
    save_ov_model_from_optimum(opt_model, hf_tokenizer, model_path)

    cb_pipe = ContinuousBatchingPipeline(model_path, Tokenizer(model_path), scheduler_config, "CPU")

    outputs = cb_pipe.generate(["What is OpenVINO?"], generation_configs)
    assert(len(outputs))
    for output in outputs:
        assert(len(output.m_generation_ids))
        for sequence in output.m_generation_ids:
            assert(sequence == "What is OpenVINO?")


@pytest.mark.precommit
@pytest.mark.parametrize("get_generation_config", [get_greedy, get_beam_search, get_multinomial_all_parameters],
                         ids=["greedy", "beam_search", "multinomial_all_parameters"])
@pytest.mark.parametrize("max_num_batched_tokens", [2, 4, 256])
def test_echo_with_generation_phase(tmp_path, get_generation_config, max_num_batched_tokens):
    generation_config = get_generation_config()
    generation_config.max_new_tokens = 10
    generation_config.echo = True

    scheduler_config = get_scheduler_config()
    scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    generation_configs = [generation_config]
    model_id : str = "facebook/opt-125m"
    opt_model, hf_tokenizer = get_hugging_face_model_and_tokenizer(model_id, use_optimum=True)

    model_path : Path = tmp_path / model_id
    save_ov_model_from_optimum(opt_model, hf_tokenizer, model_path)

    cb_pipe = ContinuousBatchingPipeline(model_path, Tokenizer(model_path), scheduler_config, "CPU")

    outputs = cb_pipe.generate(["What is OpenVINO?"], generation_configs)
    assert(len(outputs))
    for output in outputs:
        assert(len(output.m_generation_ids))
        for sequence in output.m_generation_ids:
            assert(sequence.startswith("What is OpenVINO?"))
            assert(len(sequence) > len("What is OpenVINO?"))
