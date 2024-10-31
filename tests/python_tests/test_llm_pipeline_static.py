# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_genai as ov_genai
from openvino.runtime import Core
import pytest
import sys
from ov_genai_test_utils import (
    get_models_list,
    get_chat_models_list,
)


# This test suite is designed specifically to validate the functionality and robustness of the StaticLLMPipeline on NPUW:CPU.
common_config = {
                      'NPU_USE_NPUW': 'YES',
                      'NPUW_DEVICES': 'CPU',
                      'NPUW_ONLINE_PIPELINE': 'NONE',
                      'PREFILL_CONFIG': { },
                      'GENERATE_CONFIG': { }
                }


def generate_chat_history(model_path, device, pipeline_config, questions):
    pipe = ov_genai.LLMPipeline(model_path, device, **pipeline_config)
    pipe.start_chat()
    chat_history = [ pipe.generate(question, max_new_tokens=50) for question in questions ]
    pipe.finish_chat()
    return chat_history


@pytest.mark.skipif(sys.platform in ["darwin", "linux"], reason="Not supposed to work on mac. Segfault on linux CI")
@pytest.mark.precommit
@pytest.mark.nightly
def test_generation_compare_with_stateful():
    prompt = 'The Sun is yellow because'
    model_path = get_models_list()[0][1]

    stateful_pipe = ov_genai.LLMPipeline(model_path, "CPU")
    ref_out = stateful_pipe.generate(prompt, max_new_tokens=100)

    static_pipe = ov_genai.LLMPipeline(model_path, "NPU", **common_config)
    actual_out = static_pipe.generate(prompt, max_new_tokens=100)

    if ref_out != actual_out:
        print(f'ref_out: {ref_out}\n')
        print(f'actual_out: {actual_out}')
    assert ref_out == actual_out


@pytest.mark.skipif(sys.platform in ["darwin", "linux"], reason="Not supposed to work on mac. Segfault on linux CI")
@pytest.mark.precommit
@pytest.mark.nightly
def test_length_properties_set_no_exception():
    model_path = get_models_list()[0][1]
    # NB: Check it doesn't throw any exception
    pipeline_config = { "MAX_PROMPT_LEN": 128, "MIN_RESPONSE_LEN": 64 }
    pipeline_config |= common_config
    pipe = ov_genai.LLMPipeline(model_path, "NPU", **pipeline_config)


pipeline_configs = [
    { "MAX_PROMPT_LEN":   -1  },
    { "MAX_PROMPT_LEN":   "1" },
    { "MIN_RESPONSE_LEN": -1  },
    { "MIN_RESPONSE_LEN": "1" }
]
@pytest.mark.skipif(sys.platform in ["darwin", "linux"], reason="Not supposed to work on mac. Segfault on linux CI")
@pytest.mark.parametrize("pipeline_config", pipeline_configs)
@pytest.mark.precommit
@pytest.mark.nightly
def test_invalid_length_properties_raise_error(pipeline_config):
    model_path = get_models_list()[0][1]
    pipeline_config |= common_config
    with pytest.raises(RuntimeError):
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **pipeline_config)


@pytest.mark.skipif(sys.platform in ["darwin", "linux"], reason="Not supposed to work on mac. Segfault on linux CI")
@pytest.mark.precommit
@pytest.mark.nightly
def test_batch_one_no_exception():
    model_path = get_models_list()[0][1]
    prompt = 'The Sun is yellow because'
    static_pipe = ov_genai.LLMPipeline(model_path, "NPU", **common_config)
    # Check it doesn't throw any exception when batch of size 1 is provided
    actual_out = static_pipe.generate([prompt], max_new_tokens=20)


# TODO: For the further batch support
@pytest.mark.skipif(sys.platform in ["darwin", "linux"], reason="Not supposed to work on mac. Segfault on linux CI")
@pytest.mark.precommit
@pytest.mark.nightly
def test_batch_raise_error():
    model_path = get_models_list()[0][1]
    prompt = 'The Sun is yellow because'
    pipe = ov_genai.LLMPipeline(model_path, "NPU", **common_config)
    with pytest.raises(RuntimeError):
        pipe.generate([prompt] * 3, max_new_tokens=100)


# TODO: For the further sampling support
generation_configs = [
    dict(num_beam_groups=3),
    dict(do_sample=True)
]
@pytest.mark.skipif(sys.platform in ["darwin", "linux"], reason="Not supposed to work on mac. Segfault on linux CI")
@pytest.mark.parametrize("generation_config", generation_configs)
@pytest.mark.precommit
@pytest.mark.nightly
def test_unsupported_sampling_raise_error(generation_config):
    model_path = get_models_list()[0][1]
    prompt = 'The Sun is yellow because'
    pipe = ov_genai.LLMPipeline(model_path, "NPU", **common_config)
    with pytest.raises(RuntimeError):
        pipe.generate(prompt, **generation_config)


@pytest.mark.skipif(sys.platform in ["darwin", "linux"], reason="Not supposed to work on mac. Segfault on linux CI")
@pytest.mark.precommit
@pytest.mark.nightly
def test_max_number_of_tokens():
    model_path = get_models_list()[0][1]
    prompt = 'The Sun is yellow because'
    num_tokens = 128

    pipe = ov_genai.LLMPipeline(model_path, "NPU", **common_config)
    tokenizer = ov_genai.Tokenizer(model_path)
    tokenized_input = tokenizer.encode(prompt)
    # ignore_eos=True to ensure model will generate exactly num_tokens
    encoded_results = pipe.generate(tokenized_input, max_new_tokens=num_tokens, ignore_eos=True)
    assert len(encoded_results.tokens[0]) == num_tokens


@pytest.mark.skipif(sys.platform in ["darwin", "linux"], reason="Not supposed to work on mac. Segfault on linux CI")
@pytest.mark.precommit
@pytest.mark.nightly
def test_chat_generation(model_descr):
    questions = [
        '1+1=',
        'What is the previous answer?',
        'Why is the Sun yellow?',
        'What was my first question?'
    ]

    model_path = get_chat_models_lists()[0][1]

    chat_history_stateful = generate_chat_history(model_path, "CPU", { }, questions)
    chat_history_static   = generate_chat_history(model_path, "NPU", common_config, questions)

    print('npu chat: \n{chat_history_static}\n')
    print('cpu chat: \n{chat_history_stateful}')

    if chat_history_stateful != chat_history_static:
        print(f'hf_output: {chat_history_static}')
        print(f'ov_output: {chat_history_stateful}')
    assert chat_history_stateful == chat_history_static
