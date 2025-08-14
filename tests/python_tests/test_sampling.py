# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from openvino_genai import GenerationConfig, StopCriteria

from utils.ov_genai_pipelines import generate_and_compare, run_ov_pipeline, get_main_pipeline_types
from utils.hugging_face import download_and_convert_model

@pytest.mark.precommit
@pytest.mark.parametrize("generation_config,prompt",
                         [(dict(max_new_tokens=30), 'table is made of'),
                          (dict(max_new_tokens=30, min_new_tokens=30), '你好！ 你好嗎？'),
                          (dict(max_new_tokens=30, ignore_eos=True), 'Alan Turing was a'),
                          (dict(max_length=30, ignore_eos=True), 'table is made of'),
                          (dict(stop_token_ids={28998}, apply_chat_template=False), 'The Sun is yellow because'), # since a test does not hang, it means stop token is met, skip chat template to generate long answer
                        #   (dict(max_new_tokens=1, min_new_tokens=0, echo=True), 'What is OpenVINO?')
                          ],
                         ids=["max_new_tokens",
                              "min_and_max_new_tokens",
                              "max_new_tokens_and_ignore_eos_true",
                              "max_length",
                              "stop_token_ids",
                            #   "echo_with_generation",
                              ])
def test_basic_stop_criteria(generation_config, prompt):
    model_id : str = "katuni4ka/tiny-random-phi3"
    generate_and_compare(model_id, [prompt], generation_config)


@pytest.mark.precommit
@pytest.mark.parametrize("generation_config,model_id",
                         [(dict(max_new_tokens=50, min_new_tokens=15, stop_strings={"anag"}, include_stop_str_in_output=True), 'facebook/opt-125m'), # expected match on "manage"
                          (dict(max_new_tokens=50, min_new_tokens=1, stop_strings={".", "software", "Intel"}, include_stop_str_in_output=True), 'facebook/opt-125m'),
                          (dict(max_new_tokens=50, min_new_tokens=1, stop_strings={"Einstein", "sunny", "geothermal"}, include_stop_str_in_output=True), 'facebook/opt-125m'), # expected no match
                          (dict(max_new_tokens=30, stop_strings={ "machines" }, include_stop_str_in_output=False),'facebook/opt-125m'),
                          (dict(max_new_tokens=30, stop_strings={ "machines" }, include_stop_str_in_output=True), 'facebook/opt-125m'),
                          (dict(max_new_tokens=30, stop_strings={ "machines", "manage" }, include_stop_str_in_output=False), 'facebook/opt-125m'),
                          (dict(max_new_tokens=30, stop_strings={ "machines", "manage" }, include_stop_str_in_output=True), 'facebook/opt-125m'),
                          (dict(max_new_tokens=30, stop_strings={ "software toolkit developed 1 by", "Intel" }, include_stop_str_in_output=False), 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')],
                         ids=["single_stop_string",
                              "multiple_stop_strings_match",
                              "multiple_stop_strings_no_match",
                              "single_stop_string_exclude_from_output",
                              "single_stop_string_include_to_output",
                              "multiple_stop_strings_exclude_from_output",
                              "multiple_stop_strings_include_to_output",
                              "multiple_stop_strings_one_no_match_and_long_exclude_from_output"])
@pytest.mark.parametrize("pipeline_type", get_main_pipeline_types())
def test_stop_strings(generation_config, model_id, pipeline_type):
    prompts = [ "What is OpenVINO?" ]
    generate_and_compare(model_id, prompts, generation_config, pipeline_type=pipeline_type)


@pytest.mark.precommit
@pytest.mark.parametrize("generation_config",
                         [dict(max_new_tokens=30),
                          dict(max_new_tokens=30, repetition_penalty=2.0),
                          dict(max_new_tokens=300, apply_chat_template=False)],
                         ids=["basic", "repetition_penalty", "long_max_new_tokens"])
@pytest.mark.parametrize("prompt", [
    'What is OpenVINO?',
    'table is made of', 
    'The Sun is yellow because', 
    '你好！ 你好嗎？'.encode('unicode_escape'),  # to escape Win limitation on Unicode tmp path
    'I have an interview about product speccing with the company Weekend Health. Give me an example of a question they might ask with regards about a new feature'
])
def test_greedy(generation_config, prompt):
    model_id : str = "katuni4ka/tiny-random-phi3"
    prompt = prompt.decode('unicode_escape') if isinstance(prompt, bytes) else prompt

    generate_and_compare(model=model_id, 
                         prompts=prompt, 
                         generation_config=generation_config)


@pytest.mark.precommit
@pytest.mark.parametrize("generation_config",
                         [dict(max_new_tokens=30, num_beams=2),
                          dict(max_new_tokens=30, num_beams=2, stop_criteria=StopCriteria.NEVER),
                          dict(max_new_tokens=30, num_beams=2, stop_criteria=StopCriteria.EARLY),
                        #   dict(max_new_tokens=30, num_beams=2, echo=True),
                          dict(max_new_tokens=30, num_beams=2, length_penalty=1.0),
                          dict(max_new_tokens=30, num_beams=2, no_repeat_ngram_size=2),
                          dict(max_new_tokens=30, num_beams=6, num_beam_groups=3, diversity_penalty=1.2, num_return_sequences=3),
                          dict(max_new_tokens=30, min_new_tokens=15, num_beams=2, num_return_sequences=1),
                          dict(max_new_tokens=30, num_beams=2, stop_strings={"Einstein", "sunny", "geothermal"}, include_stop_str_in_output=True),],
                         ids=["single_group_stop_criteria_heuristic",
                              "single_group_stop_criteria_never",
                              "single_group_stop_criteria_early",
                            #   "single_group_with_echo",
                              "single_group_lenght_penalty",
                              "single_group_no_repeat_ngram_size",
                              "multiple_groups",
                              "single_group_min_new_tokens",
                              "single_group_with_multiple_stop_strings_no_match",])
def test_beam_search(generation_config):
    prompts = [ "What is OpenVINO?" ]
    model_id : str = "facebook/opt-125m"
    generate_and_compare(model_id, prompts, generation_config)


@pytest.mark.precommit
@pytest.mark.xfail(
    raises=AssertionError,
    reason="Stop strings do not seem to work as expected with beam search in HF, so comparison will fail. If it changes, these cases shall be merged to the test above.",
    strict=True,
)
@pytest.mark.parametrize("generation_config",
                         [dict(max_new_tokens=50, num_beams=6, num_beam_groups=3, diversity_penalty=1.0, num_return_sequences=6, stop_strings={"open sour"}, include_stop_str_in_output=True),
                          dict(max_new_tokens=50, num_beams=6, num_beam_groups=3, diversity_penalty=1.0, num_return_sequences=6, stop_strings={".", "software", "Intel"}, include_stop_str_in_output=True),],
                         ids=["single_stop_string_match", "multiple_stop_strings_match"])
def test_beam_search_with_stop_string(generation_config):
    prompts = [ "What is OpenVINO?" ]
    model_id : str = "facebook/opt-125m"
    generate_and_compare(model_id, prompts, generation_config)


@pytest.mark.precommit
@pytest.mark.parametrize("generation_config",
                         [dict(max_new_tokens=1, min_new_tokens=0, echo=True),
                          dict(max_new_tokens=30, num_beams=2, echo=True),],
                         ids=["echo_with_generation",
                              "single_group_with_echo",])
def test_echo(generation_config):
    prompts = [ "What is OpenVINO?" ]
    model_id : str = "facebook/opt-125m"
    # TODO: support in stateful mode and remove 'use_cb=True' and this test at all
    # as we can enable new parameters set in other tests
    generate_and_compare(model_id, prompts, generation_config)


# TODO: remove platform specific reference texts once CVS-159912 is done and use comparison with HF
# and merge this tests with 'test_sampling_against_optimum' by extending a list of generation configs

class PlatformsRefTexts(TypedDict, total=False):
    linux: list[list[str]]
    win32: list[list[str]]
    darwin: list[list[str]]


def get_current_platform_ref_texts(ref_texts: PlatformsRefTexts) -> list[list[str]]:
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
    prompts: list[str]
    ref_texts: list[list[str]]

from utils.generation_config import get_multinomial_temperature, get_greedy_with_penalties, \
    get_multinomial_temperature_and_top_k, get_multinomial_temperature_and_top_p, \
    get_multinomial_temperature_top_p_and_top_k, get_multinomial_all_parameters, \
    get_multinomial_temperature_and_num_return_sequence, get_multinomial_max_and_min_token, \
    get_multinomial_temperature_and_frequence_penalty, get_multinomial_temperature_and_presence_penalty, \
    get_multinomial_temperature_and_repetition_penalty

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
        generation_config=get_greedy_with_penalties(),
        prompts=["What is OpenVINO?"],
        ref_texts=[
            [
                "\nOpenVINO is a software that allows users to create and manage their own virtual machines. It's designed for use with Windows, Mac OS X"
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
             "greedy_with_penalties",
             "multinomial_max_and_min_token"])
def test_multinomial_sampling_against_reference(test_struct: RandomSamplingTestStruct):
    generation_config = test_struct.generation_config

    prompts = test_struct.prompts
    generation_config.rng_seed = 0

    model_id : str = "facebook/opt-125m"
    _, _, models_path = download_and_convert_model(model_id)

    # Run multinomial without comparison with HF reference.
    _ = run_ov_pipeline(models_path=models_path,
                        prompt=prompts,
                        generation_config=generation_config)

    # Reference comparison is not performed as sampling results are non-deterministic.
    # Discrete_distribution impl depends on platform, model inference results may depend on CPU.
