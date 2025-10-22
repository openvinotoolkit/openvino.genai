# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino_genai import Tokenizer, LLMPipeline, StreamerBase, ChatHistory
import os

import pytest
import platform
import sys

from utils.constants import get_default_llm_properties
from utils.tokenizers import model_tmp_path
from utils.hugging_face import download_and_convert_model
from utils.ov_genai_pipelines import create_ov_pipeline
from utils.generation_config import                     \
    get_greedy,                                         \
    get_greedy_with_penalties,                          \
    get_multinomial_all_parameters,                     \
    get_multinomial_temperature_and_presence_penalty,   \
    get_beam_search
from data.models import get_models_list, get_chat_models_list


if sys.platform == 'darwin' or platform.machine() in ["aarch64", "arm64", "ARM64"]:
    pytest.skip("NPU plugin is available only on Linux and Windows x86_64", allow_module_level=True)


default_config = {
    'NPUW_DEVICES': 'CPU',
    'NPUW_ONLINE_PIPELINE': 'NONE'
} | get_default_llm_properties()

static_config = { **default_config, 'STATIC_PIPELINE': 'STATEFUL' }

# Test both, static and generic pipelines
pipeline_configs = [default_config, static_config]

blob_with_weights = [True, False]


def generate_with_chat_mode(model_path, device, pipeline_config, questions) -> list[str]:
    pipe = LLMPipeline(model_path, device, **pipeline_config)
    pipe.start_chat()
    answers = [ pipe.generate(question, max_new_tokens=50, do_sample=False) for question in questions ]
    pipe.finish_chat()
    return answers


def generate_with_chat_history(model_path, device, pipeline_config, questions) -> ChatHistory:
    pipe = LLMPipeline(model_path, device, **pipeline_config)
    chat_history = ChatHistory()
    for question in questions:
        chat_history.append({"role": "user", "content": question})
        decoded_results = pipe.generate(chat_history, max_new_tokens=50, do_sample=False)
        chat_history.append({"role": "assistant", "content": decoded_results.texts[0]})                          
    return chat_history


prompt = 'What is OpenVINO?'
generate_inputs = [
    [prompt],
    ChatHistory([{ "role": "user", "content": prompt }])
]
generation_configs = [
    get_greedy(),
    get_greedy_with_penalties()
]
@pytest.mark.precommit
@pytest.mark.parametrize("generation_config", generation_configs)
@pytest.mark.parametrize("config", pipeline_configs)
@pytest.mark.parametrize("model_id", get_models_list())
@pytest.mark.parametrize("input", generate_inputs)
@pytest.mark.xfail(reason="Generation result mismatch. Ticket 171117", raises=AssertionError)
def test_generation_compare_with_stateful(generation_config, config, model_id, input):
    _, _, model_path = download_and_convert_model(model_id)

    stateful_pipe = LLMPipeline(model_path, "CPU", **get_default_llm_properties())
    ref_out = stateful_pipe.generate(input, generation_config)

    static_pipe = LLMPipeline(model_path, "NPU", **config)
    actual_out = static_pipe.generate(input, generation_config)

    assert ref_out.texts[0] == actual_out.texts[0]


@pytest.mark.precommit
@pytest.mark.parametrize("config", pipeline_configs)
@pytest.mark.parametrize("with_weights", blob_with_weights)
@pytest.mark.parametrize("model_id", get_models_list())
def test_pipeline_from_blob(model_tmp_path, config, with_weights, model_id):
    prompt = 'What is OpenVINO?'
    _, _, model_path = download_and_convert_model(model_id)
    _, temp_path = model_tmp_path

    blob_path = os.path.join(temp_path, "compiled_model.blob")

    cpu_pipe = LLMPipeline(model_path, "CPU", **get_default_llm_properties())
    ref_out = cpu_pipe.generate(prompt, max_new_tokens=30)

    # NB: Generate the blob
    cfg = { "EXPORT_BLOB": "YES", "BLOB_PATH": blob_path }
    cfg |= config
    if with_weights:
        cfg |= {"CACHE_MODE": "OPTIMIZE_SPEED"}
    npu_pipe = LLMPipeline(model_path, "NPU", **cfg)
    actual_out = npu_pipe.generate(prompt, max_new_tokens=30)
    assert ref_out == actual_out
    del npu_pipe

    # Import blob and check accuracy
    import_cfg = {"BLOB_PATH": blob_path, "WEIGHTS_PATH": os.path.join(model_path, "openvino_model.bin") }
    import_cfg |= config
    if with_weights:
        import_cfg.pop("WEIGHTS_PATH")
    npu_pipe = LLMPipeline(model_path, "NPU", **import_cfg)
    actual_out = npu_pipe.generate(prompt, max_new_tokens=30)

    assert ref_out == actual_out


@pytest.mark.precommit
@pytest.mark.parametrize("config", pipeline_configs)
@pytest.mark.parametrize("with_weights", blob_with_weights)
@pytest.mark.parametrize("model_id", get_models_list())
def test_pipeline_cache_dir(model_tmp_path, config, with_weights, model_id):
    prompt = 'What is OpenVINO?'
    _, _, model_path = download_and_convert_model(model_id)
    _, temp_path = model_tmp_path

    cpu_pipe = LLMPipeline(model_path, "CPU", **get_default_llm_properties())
    ref_out = cpu_pipe.generate(prompt, max_new_tokens=30)

    # NB: Generate the blob
    cfg = { "NPUW_DEVICES": "CPU", "CACHE_DIR": str(temp_path) }
    cfg |= config
    if with_weights:
        cfg |= {"CACHE_MODE": "OPTIMIZE_SPEED"}
    npu_pipe = LLMPipeline(model_path, "NPU", **cfg)
    actual_out = npu_pipe.generate(prompt, max_new_tokens=30)
    assert ref_out == actual_out
    del npu_pipe

    # Check that blob was cached
    blobs = [file for file in os.listdir(temp_path) if file.endswith(".blob")]
    if len(blobs) == 0:
        print(f"Couldn't cache the blob")
    assert len(blobs) > 0

    # Import blob and check accuracy
    npu_pipe = LLMPipeline(model_path, "NPU", **(config | { "CACHE_DIR": str(temp_path) }))
    actual_out = npu_pipe.generate(prompt, max_new_tokens=30)

    # Check that blob was used from cache
    blobs = [file for file in os.listdir(temp_path) if file.endswith(".blob")]
    if len(blobs) == 0:
        print(f"Couldn't cache the blob")
    assert len(blobs) > 0

    assert ref_out == actual_out


generation_configs = [
    get_multinomial_temperature_and_presence_penalty()
]
@pytest.mark.precommit
@pytest.mark.parametrize("generation_config", generation_configs)
@pytest.mark.parametrize("config", pipeline_configs)
@pytest.mark.parametrize("model_id", get_models_list())
def test_multinomial_sampling(generation_config, config, model_id):
    # Multinomial sampling is highly sensitive to raw logits values. For fair comparison,
    # a reference implementation producing identical logits (e.g., from StaticLLMPipeline)
    # would be necessary. However, the CPU in StatefulPipeline and StaticLLMPipeline may apply
    # different optimizations due to differences in provided topologies, leading to slight
    # variations in raw logits. Therefore, there is no reliable reference for validation,
    # so only ensure that no exceptions are raised.
    prompt = 'What is OpenVINO?'
    _, _, model_path = download_and_convert_model(model_id)
    static_pipe = LLMPipeline(model_path, "NPU", **config)
    actual_out = static_pipe.generate(prompt, generation_config)


@pytest.mark.precommit
@pytest.mark.parametrize("config", pipeline_configs)
@pytest.mark.parametrize("model_id", get_models_list())
def test_length_properties_set_no_exception(config, model_id):
    _, _, model_path = download_and_convert_model(model_id)
    # NB: Check it doesn't throw any exception
    pipeline_config = { "MAX_PROMPT_LEN": 256, "MIN_RESPONSE_LEN": 64 }
    pipeline_config |= config
    pipe = LLMPipeline(model_path, "NPU", **pipeline_config)


length_configs = [
    { "MAX_PROMPT_LEN":   -1  },
    { "MAX_PROMPT_LEN":   "1" },
    { "MIN_RESPONSE_LEN": -1  },
    { "MIN_RESPONSE_LEN": "1" }
]
@pytest.mark.parametrize("length_config", length_configs)
@pytest.mark.parametrize("config", pipeline_configs)
@pytest.mark.parametrize("model_id", get_models_list())
@pytest.mark.precommit
def test_invalid_length_properties_raise_error(length_config, config, model_id):
    _, _, model_path = download_and_convert_model(model_id)
    length_config |= config
    with pytest.raises(RuntimeError):
        pipe = LLMPipeline(model_path, "NPU", **length_config)


@pytest.mark.precommit
@pytest.mark.parametrize("config", pipeline_configs)
@pytest.mark.parametrize("model_id", get_models_list())
def test_batch_one_no_exception(config, model_id):
    _, _, model_path = download_and_convert_model(model_id)
    prompt = 'The Sun is yellow because'
    static_pipe = LLMPipeline(model_path, "NPU", **config)
    # Check it doesn't throw any exception when batch of size 1 is provided
    actual_out = static_pipe.generate([prompt], max_new_tokens=20)


# TODO: For the further batch support
@pytest.mark.precommit
@pytest.mark.parametrize("config", pipeline_configs)
@pytest.mark.parametrize("model_id", get_models_list())
def test_batch_raise_error(config, model_id):
    _, _, model_path = download_and_convert_model(model_id)
    prompt = 'The Sun is yellow because'
    pipe = LLMPipeline(model_path, "NPU", **config)
    with pytest.raises(RuntimeError):
        pipe.generate([prompt] * 3, max_new_tokens=100)


# TODO: For the further sampling support
generation_configs = [
    get_beam_search(),
    # NB: Only num_return_sequences=1 is supported!
    get_multinomial_all_parameters()
]
@pytest.mark.parametrize("generation_config", generation_configs)
@pytest.mark.parametrize("config", pipeline_configs)
@pytest.mark.parametrize("model_id", get_models_list())
@pytest.mark.precommit
def test_unsupported_sampling_raise_error(generation_config, config, model_id):
    _, _, model_path = download_and_convert_model(model_id)
    prompt = 'What is OpenVINO?'

    pipe = LLMPipeline(model_path, "NPU", **config)
    with pytest.raises(RuntimeError):
        pipe.generate(prompt, generation_config)


@pytest.mark.precommit
@pytest.mark.parametrize("config", pipeline_configs)
@pytest.mark.parametrize("model_id", get_models_list())
def test_terminate_by_max_number_of_tokens(config, model_id):
    _, _, model_path = download_and_convert_model(model_id)
    prompt = 'The Sun is yellow because'
    num_tokens = 128

    pipe = LLMPipeline(model_path, "NPU", **config)
    tokenizer = Tokenizer(model_path)
    tokenized_input = tokenizer.encode(prompt)
    # ignore_eos=True to ensure model will generate exactly num_tokens
    encoded_results = pipe.generate(tokenized_input, max_new_tokens=num_tokens, ignore_eos=True)
    assert len(encoded_results.tokens[0]) == num_tokens


@pytest.mark.precommit
@pytest.mark.parametrize("config", pipeline_configs)
@pytest.mark.parametrize("model_id", get_models_list())
def test_terminate_by_out_of_memory(config, model_id):
    _, _, model_path = download_and_convert_model(model_id)
    prompt = 'The Sun is yellow because'
    pipeline_config = { "MAX_PROMPT_LEN": 256, "MIN_RESPONSE_LEN": 64 }
    pipeline_config |= config
    kv_cache_size = pipeline_config['MAX_PROMPT_LEN'] + pipeline_config['MIN_RESPONSE_LEN']

    tokenizer = Tokenizer(model_path)
    tokenized_input = tokenizer.encode(prompt)
    input_len = tokenized_input.input_ids.get_shape()[1]

    pipe = LLMPipeline(model_path, "NPU", **pipeline_config)
    encoded_results = pipe.generate(tokenized_input, max_new_tokens=1000, ignore_eos=True)

    assert len(encoded_results.tokens[0]) == (kv_cache_size - input_len + 1)


@pytest.mark.precommit
@pytest.mark.parametrize("config", pipeline_configs)
@pytest.mark.parametrize("model_id", get_models_list())
def test_terminate_by_sampler(config, model_id):
    _, _, model_path = download_and_convert_model(model_id)
    prompt = 'The Sun is yellow because'

    current_iter = 0
    num_iters = 10

    class TestStreamer(StreamerBase):
        def __init__(self):
            StreamerBase.__init__(self)
        def put(self, token_id):
            nonlocal current_iter
            current_iter += 1
            return current_iter == num_iters
        def end(self):
            pass

    tokenizer = Tokenizer(model_path)
    tokenized_input = tokenizer.encode(prompt)

    pipe = LLMPipeline(model_path, "NPU", **config)
    encoded_results = pipe.generate(tokenized_input, max_new_tokens=1000, ignore_eos=True, streamer=TestStreamer())

    assert len(encoded_results.tokens[0]) == num_iters


# FIXME: Known problem, output differs from stateful pipeline starting from 3rd prompt!
@pytest.mark.skip(reason="JIRA-144780: Output differs from stateful pipeline")
@pytest.mark.parametrize("config", pipeline_configs)
@pytest.mark.parametrize("model_id", get_models_list())
@pytest.mark.precommit
def test_chat_generation(config, model_id):
    questions = [
        '1+1=',
        'What is the previous answer?',
        'Why is the Sun yellow?',
        'What was my first question?'
    ]

    _, _, model_path = download_and_convert_model(model_id)

    answers_chat_mode_stateful = generate_with_chat_mode(model_path, "CPU", get_default_llm_properties(), questions)
    answers_chat_mode_static = generate_with_chat_mode(model_path, "NPU", config, questions)
    assert answers_chat_mode_stateful == answers_chat_mode_static, \
        f"CPU output:\n{answers_chat_mode_stateful}\nNPU output:\n{answers_chat_mode_static}"

    chat_history_stateful = generate_with_chat_history(model_path, "CPU", get_default_llm_properties(), questions)
    messages_stateful = chat_history_stateful.get_messages()
    chat_history_static = generate_with_chat_history(model_path, "NPU", config, questions)
    messages_static = chat_history_static.get_messages()
    assert messages_stateful == messages_static, f"CPU output:\n{messages_stateful}\nNPU output:\n{messages_static}"
    
    answers_chat_history_static = [msg["content"] for msg in messages_static if msg["role"] == "assistant"]
    assert answers_chat_mode_static == answers_chat_history_static, (
        f"NPU chat mode output:\n{answers_chat_mode_static}\n"
        f"NPU chat history output:\n{answers_chat_history_static}"
    )
