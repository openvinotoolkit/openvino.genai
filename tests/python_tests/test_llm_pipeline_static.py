# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_genai as ov_genai
from openvino_genai import GenerationConfig

import pytest
import platform
import sys
from ov_genai_test_utils import (
    get_models_list,
    get_chat_models_list,
    read_model
)
from utils.constants import get_default_llm_properties
from utils.generation_config import                     \
    get_greedy,                                         \
    get_greedy_with_penalties,                          \
    get_multinomial_all_parameters,                     \
    get_multinomial_temperature_and_presence_penalty,   \
    get_beam_search


if sys.platform == 'darwin' or platform.machine() in ["aarch64", "arm64", "ARM64"]:
    pytest.skip("NPU plugin is available only on Linux and Windows x86_64", allow_module_level=True)


# This test suite is designed specifically to validate the functionality and robustness of the StaticLLMPipeline on NPUW:CPU.
common_config = {
                      'NPU_USE_NPUW': 'YES',
                      'NPUW_DEVICES': 'CPU',
                      'NPUW_ONLINE_PIPELINE': 'NONE',
                      'PREFILL_CONFIG': { },
                      'GENERATE_CONFIG': { }
                } | get_default_llm_properties()


def generate_chat_history(model_path, device, pipeline_config, questions):
    pipe = ov_genai.LLMPipeline(model_path, device, **pipeline_config)
    pipe.start_chat()
    chat_history = [ pipe.generate(question, max_new_tokens=50, do_sample=False) for question in questions ]
    pipe.finish_chat()
    return chat_history


generation_configs = [
    get_greedy(),
    get_greedy_with_penalties()
]
@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("generation_config", generation_configs)
def test_generation_compare_with_stateful(generation_config):
    prompt = 'What is OpenVINO?'
    model_path = read_model(get_models_list()[0])[1]

    stateful_pipe = ov_genai.LLMPipeline(model_path, "CPU", **get_default_llm_properties())
    ref_out = stateful_pipe.generate(prompt, generation_config)

    static_pipe = ov_genai.LLMPipeline(model_path, "NPU", **common_config)
    actual_out = static_pipe.generate(prompt, generation_config)

    if ref_out != actual_out:
        print(f'ref_out: {ref_out}\n')
        print(f'actual_out: {actual_out}')
    assert ref_out == actual_out


generation_configs = [
    get_multinomial_temperature_and_presence_penalty()
]
@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("generation_config", generation_configs)
def test_multinomial_sampling(generation_config):
    # Multinomial sampling is highly sensitive to raw logits values. For fair comparison,
    # a reference implementation producing identical logits (e.g., from StaticLLMPipeline)
    # would be necessary. However, the CPU in StatefulPipeline and StaticLLMPipeline may apply
    # different optimizations due to differences in provided topologies, leading to slight
    # variations in raw logits. Therefore, there is no reliable reference for validation,
    # so only ensure that no exceptions are raised.
    prompt = 'What is OpenVINO?'
    model_path = read_model(get_models_list()[0])[1]
    static_pipe = ov_genai.LLMPipeline(model_path, "NPU", **common_config)
    actual_out = static_pipe.generate(prompt, generation_config)


@pytest.mark.precommit
@pytest.mark.nightly
def test_length_properties_set_no_exception():
    model_path = read_model(get_models_list()[0])[1]
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
@pytest.mark.parametrize("pipeline_config", pipeline_configs)
@pytest.mark.precommit
@pytest.mark.nightly
def test_invalid_length_properties_raise_error(pipeline_config):
    model_path = read_model(get_models_list()[0])[1]
    pipeline_config |= common_config
    with pytest.raises(RuntimeError):
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **pipeline_config)


@pytest.mark.precommit
@pytest.mark.nightly
def test_batch_one_no_exception():
    model_path = read_model(get_models_list()[0])[1]
    prompt = 'The Sun is yellow because'
    static_pipe = ov_genai.LLMPipeline(model_path, "NPU", **common_config)
    # Check it doesn't throw any exception when batch of size 1 is provided
    actual_out = static_pipe.generate([prompt], max_new_tokens=20)


# TODO: For the further batch support
@pytest.mark.precommit
@pytest.mark.nightly
def test_batch_raise_error():
    model_path = read_model(get_models_list()[0])[1]
    prompt = 'The Sun is yellow because'
    pipe = ov_genai.LLMPipeline(model_path, "NPU", **common_config)
    with pytest.raises(RuntimeError):
        pipe.generate([prompt] * 3, max_new_tokens=100)


# TODO: For the further sampling support
generation_configs = [
    get_beam_search(),
    # NB: Only num_return_sequences=1 is supported!
    get_multinomial_all_parameters()
]
@pytest.mark.parametrize("generation_config", generation_configs)
@pytest.mark.precommit
@pytest.mark.nightly
def test_unsupported_sampling_raise_error(generation_config):
    model_path = read_model(get_models_list()[0])[1]
    prompt = 'What is OpenVINO?'

    pipe = ov_genai.LLMPipeline(model_path, "NPU", **common_config)
    with pytest.raises(RuntimeError):
        pipe.generate(prompt, generation_config)


@pytest.mark.precommit
@pytest.mark.nightly
def test_terminate_by_max_number_of_tokens():
    model_path = read_model(get_models_list()[0])[1]
    prompt = 'The Sun is yellow because'
    num_tokens = 128

    pipe = ov_genai.LLMPipeline(model_path, "NPU", **common_config)
    tokenizer = ov_genai.Tokenizer(model_path)
    tokenized_input = tokenizer.encode(prompt)
    # ignore_eos=True to ensure model will generate exactly num_tokens
    encoded_results = pipe.generate(tokenized_input, max_new_tokens=num_tokens, ignore_eos=True)
    assert len(encoded_results.tokens[0]) == num_tokens


@pytest.mark.precommit
@pytest.mark.nightly
def test_terminate_by_out_of_memory():
    model_path = read_model(get_models_list()[0])[1]
    prompt = 'The Sun is yellow because'
    pipeline_config = { "MAX_PROMPT_LEN": 64, "MIN_RESPONSE_LEN": 64 }
    pipeline_config |= common_config
    kv_cache_size = pipeline_config['MAX_PROMPT_LEN'] + pipeline_config['MIN_RESPONSE_LEN']

    tokenizer = ov_genai.Tokenizer(model_path)
    tokenized_input = tokenizer.encode(prompt)
    input_len = tokenized_input.input_ids.get_shape()[1]

    pipe = ov_genai.LLMPipeline(model_path, "NPU", **pipeline_config)
    encoded_results = pipe.generate(tokenized_input, max_new_tokens=1000, ignore_eos=True)

    assert len(encoded_results.tokens[0]) == (kv_cache_size - input_len + 1)


@pytest.mark.precommit
@pytest.mark.nightly
def test_terminate_by_sampler():
    model_path = read_model(get_models_list()[0])[1]
    prompt = 'The Sun is yellow because'

    current_iter = 0
    num_iters = 10

    class TestStreamer(ov_genai.StreamerBase):
        def __init__(self):
            ov_genai.StreamerBase.__init__(self)
        def put(self, token_id):
            nonlocal current_iter
            current_iter += 1
            return current_iter == num_iters
        def end(self):
            pass

    tokenizer = ov_genai.Tokenizer(model_path)
    tokenized_input = tokenizer.encode(prompt)

    pipe = ov_genai.LLMPipeline(model_path, "NPU", **common_config)
    encoded_results = pipe.generate(tokenized_input, max_new_tokens=1000, ignore_eos=True, streamer=TestStreamer())

    assert len(encoded_results.tokens[0]) == num_iters


# FIXME: Known problem, output differs from stateful pipeline starting from 3rd prompt!
@pytest.mark.skip(reason="JIRA-144780: Output differs from stateful pipeline")
@pytest.mark.precommit
@pytest.mark.nightly
def test_chat_generation():
    questions = [
        '1+1=',
        'What is the previous answer?',
        'Why is the Sun yellow?',
        'What was my first question?'
    ]

    model_path = read_model(get_chat_models_list()[0])[1]

    chat_history_stateful = generate_chat_history(model_path, "CPU", get_default_llm_properties(), questions)
    chat_history_static   = generate_chat_history(model_path, "NPU", common_config, questions)

    print('npu chat: \n{chat_history_static}\n')
    print('cpu chat: \n{chat_history_stateful}')

    if chat_history_stateful != chat_history_static:
        print(f'hf_output: {chat_history_static}')
        print(f'ov_output: {chat_history_stateful}')
    assert chat_history_stateful == chat_history_static
