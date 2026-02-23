# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino_genai import (
    GenerationConfig,
    Tokenizer,
    LLMPipeline,
    StreamerBase,
    ChatHistory,
    TokenizedInputs,
    StreamingStatus,
)
from pathlib import Path

import openvino as ov
import numpy as np
import pytest
import platform
import sys
import logging

from utils.constants import get_default_llm_properties
from utils.tokenizers import model_tmp_path
from utils.hugging_face import download_and_convert_model, OVConvertedModelSchema
from utils.generation_config import                     \
    get_greedy,                                         \
    get_greedy_with_penalties,                          \
    get_multinomial_all_parameters,                     \
    get_multinomial_temperature_and_presence_penalty,   \
    get_beam_search
from data.models import get_models_list


if sys.platform == 'darwin' or platform.machine() in ["aarch64", "arm64", "ARM64"]:
    pytest.skip("NPU plugin is available only on Linux and Windows x86_64", allow_module_level=True)


DEFAULT_CONFIG: dict = {
    'NPUW_DEVICES': 'CPU',
    'NPUW_ONLINE_PIPELINE': 'NONE'
} | get_default_llm_properties()

STATIC_CONFIG: dict  = { **DEFAULT_CONFIG, 'STATIC_PIPELINE': 'STATEFUL' }

# Test both, static and generic pipelines
PIPELINE_CONFIGS: list[dict] = [
    pytest.param(DEFAULT_CONFIG, id="generic_pipeline"), 
    pytest.param(STATIC_CONFIG, id="static_pipeline")
]

BLOB_WITH_WEIGHTS: list[bool] = [True, False]

MODELS_LIST = get_models_list()


@pytest.fixture(scope="module")
def llm_model(request: pytest.FixtureRequest) -> OVConvertedModelSchema:
    return download_and_convert_model(request.param)


@pytest.fixture(scope="module")
def ov_model(llm_model: OVConvertedModelSchema) -> LLMPipeline:
    return LLMPipeline(
        llm_model.models_path, 
        "CPU", 
        **get_default_llm_properties(),
    )


@pytest.fixture(scope="module")
def npu_config(request: pytest.FixtureRequest) -> LLMPipeline:
    return request.param


@pytest.fixture(scope="module")
def npu_model(llm_model: OVConvertedModelSchema, npu_config: dict) -> LLMPipeline:
    return LLMPipeline(
        llm_model.models_path, 
        "NPU", 
        **npu_config,
    )


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("npu_config", PIPELINE_CONFIGS, indirect=True)
@pytest.mark.parametrize("with_weights", BLOB_WITH_WEIGHTS)
def test_pipeline_from_blob(
    llm_model: OVConvertedModelSchema, 
    ov_model: LLMPipeline, 
    npu_config: dict, 
    model_tmp_path: tuple[str, Path],
    with_weights: bool
):
    prompt = 'What is OpenVINO?'
    model_path = llm_model.models_path
    _, temp_path = model_tmp_path

    blob_path = temp_path / "compiled_model.blob"

    ref_out = ov_model.generate(prompt, max_new_tokens=30)
    
    blob_path = str(blob_path)
    model_path_bin = str(model_path / "openvino_model.bin")

    # NB: Generate the blob
    cfg = { "EXPORT_BLOB": "YES", "BLOB_PATH": blob_path}
    cfg |= npu_config
    if with_weights:
        cfg |= {"CACHE_MODE": "OPTIMIZE_SPEED"}
    npu_pipe = LLMPipeline(model_path, "NPU", **cfg)
    actual_out = npu_pipe.generate(prompt, max_new_tokens=30)
    assert ref_out == actual_out
    del npu_pipe

    # Import blob and check accuracy
    import_cfg = {"BLOB_PATH": blob_path, "WEIGHTS_PATH": model_path_bin }
    import_cfg |= npu_config
    if with_weights:
        import_cfg.pop("WEIGHTS_PATH")
    npu_pipe = LLMPipeline(model_path, "NPU", **import_cfg)
    actual_out = npu_pipe.generate(prompt, max_new_tokens=30)

    assert ref_out == actual_out


@pytest.mark.parametrize(
    "generation_config", 
    [
        pytest.param(get_greedy(), id="greedy"),
        pytest.param(get_greedy_with_penalties(), id="greedy_with_penalties"),
    ]
)
@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("npu_config", PIPELINE_CONFIGS, indirect=True)
@pytest.mark.xfail(reason="Generation result mismatch. Ticket 171117", raises=AssertionError)
def test_generation_compare_with_stateful_list_input(
    ov_model: LLMPipeline, 
    npu_model: LLMPipeline,
    generation_config: GenerationConfig,
):
    input_data = ['What is OpenVINO?']
    ref_out = ov_model.generate(input_data, generation_config)
    actual_out = npu_model.generate(input_data, generation_config)

    assert ref_out.texts == actual_out.texts


@pytest.mark.parametrize(
    "generation_config", 
    [
        pytest.param(get_greedy(), id="greedy"),
        pytest.param(get_greedy_with_penalties(), id="greedy_with_penalties"),
    ]
)
@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("npu_config", PIPELINE_CONFIGS, indirect=True)
@pytest.mark.xfail(reason="Generation result mismatch. Ticket 171117", raises=AssertionError)
def test_generation_compare_with_stateful_chat_history(
    llm_model: OVConvertedModelSchema,
    npu_config: dict,
    generation_config: GenerationConfig,
):
    # ChatHistory input sets internal chat state that conflicts with other input types
    input_data = ChatHistory([{"role": "user", "content": 'What is OpenVINO?'}])
    
    ov_model_chat = LLMPipeline(llm_model.models_path, "CPU", **get_default_llm_properties())
    ref_out = ov_model_chat.generate(input_data, generation_config)
    
    npu_model_chat = LLMPipeline(llm_model.models_path, "NPU", **npu_config)
    actual_out = npu_model_chat.generate(input_data, generation_config)

    assert ref_out.texts == actual_out.texts


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("npu_config", PIPELINE_CONFIGS, indirect=True)
@pytest.mark.parametrize("with_weights", BLOB_WITH_WEIGHTS)
def test_pipeline_cache_dir(
    llm_model: OVConvertedModelSchema, 
    ov_model: LLMPipeline, 
    model_tmp_path: tuple[str, Path], 
    npu_config: dict, 
    with_weights: bool,
):
    prompt = 'What is OpenVINO?'
    model_path = llm_model.models_path
    _, temp_path = model_tmp_path
    temp_path = Path(temp_path)

    ref_out = ov_model.generate(prompt, max_new_tokens=30)

    # NB: Generate the blob
    cfg = { "NPUW_DEVICES": "CPU", "CACHE_DIR": str(temp_path) }
    cfg |= npu_config
    if with_weights:
        cfg |= {"CACHE_MODE": "OPTIMIZE_SPEED"}
    npu_pipe = LLMPipeline(model_path, "NPU", **cfg)
    actual_out = npu_pipe.generate(prompt, max_new_tokens=30)
    assert ref_out == actual_out
    del npu_pipe

    # Check that blob was cached
    blobs = [file for file in temp_path.iterdir() if file.suffix == ".blob"]    
    assert len(blobs) > 0, "Blob was not cached"

    # Import blob and check accuracy
    npu_pipe = LLMPipeline(model_path, "NPU", **(npu_config | { "CACHE_DIR": str(temp_path) }))
    actual_out = npu_pipe.generate(prompt, max_new_tokens=30)

    # Check that blob was used from cache
    blobs = [file for file in temp_path.iterdir() if file.suffix == ".blob"]    
    assert len(blobs) > 0, "Blob was not cached"

    assert ref_out == actual_out


@pytest.mark.parametrize(
    "generation_config", 
    [
        pytest.param(get_multinomial_temperature_and_presence_penalty(), id="temp+presence"),
    ]
)
@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("npu_config", PIPELINE_CONFIGS, indirect=True)
def test_multinomial_sampling(
    npu_model: LLMPipeline,
    generation_config: GenerationConfig,
):
    # Multinomial sampling is highly sensitive to raw logits values. For fair comparison,
    # a reference implementation producing identical logits (e.g., from StaticLLMPipeline)
    # would be necessary. However, the CPU in StatefulPipeline and StaticLLMPipeline may apply
    # different optimizations due to differences in provided topologies, leading to slight
    # variations in raw logits. Therefore, there is no reliable reference for validation,
    # so only ensure that no exceptions are raised.
    prompt = 'What is OpenVINO?'
    npu_model.generate(prompt, generation_config)


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("npu_config", PIPELINE_CONFIGS, indirect=True)
def test_length_properties_set_no_exception(
    llm_model: OVConvertedModelSchema, 
    npu_config: dict
):
    model_path = llm_model.models_path
    # NB: Check it doesn't throw any exception
    pipeline_config = { "MAX_PROMPT_LEN": 256, "MIN_RESPONSE_LEN": 64 }
    pipeline_config |= npu_config
    LLMPipeline(model_path, "NPU", **pipeline_config)


@pytest.mark.parametrize(
    "length_config",
    [
        { "MAX_PROMPT_LEN":   -1  },
        { "MAX_PROMPT_LEN":   "1" },
        { "MIN_RESPONSE_LEN": -1  },
        { "MIN_RESPONSE_LEN": "1" },
    ]
)
@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("npu_config", PIPELINE_CONFIGS, indirect=True)
def test_invalid_length_properties_raise_error(
    llm_model: OVConvertedModelSchema, 
    npu_config: dict,
    length_config: dict, 
):
    model_path = llm_model.models_path
    length_config |= npu_config
    with pytest.raises(RuntimeError):
        LLMPipeline(model_path, "NPU", **length_config)


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("npu_config", PIPELINE_CONFIGS, indirect=True)
def test_batch_one_no_exception(npu_model: LLMPipeline):
    prompt = 'The Sun is yellow because'
    # Check it doesn't throw any exception when batch of size 1 is provided
    npu_model.generate([prompt], max_new_tokens=20)


# TODO: For the further batch support
@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("npu_config", PIPELINE_CONFIGS, indirect=True)
def test_batch_raise_error(npu_model: LLMPipeline):
    prompt = 'The Sun is yellow because'
    with pytest.raises(RuntimeError):
        npu_model.generate([prompt] * 3, max_new_tokens=100)


# TODO: For the further sampling support
@pytest.mark.parametrize(
    "generation_config", 
    [
        pytest.param(get_beam_search(), id="beam_search"),
        # NB: Only num_return_sequences=1 is supported!
        pytest.param(get_multinomial_all_parameters(), id="multinomial")
    ]
)
@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("npu_config", PIPELINE_CONFIGS, indirect=True)
def test_unsupported_sampling_raise_error(
    npu_model: LLMPipeline, 
    generation_config: GenerationConfig, 
):
    prompt = 'What is OpenVINO?'

    with pytest.raises(RuntimeError):
        npu_model.generate(prompt, generation_config)


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("npu_config", PIPELINE_CONFIGS, indirect=True)
def test_terminate_by_max_number_of_tokens(
    llm_model: OVConvertedModelSchema,
    npu_model: LLMPipeline, 
):
    model_path = llm_model.models_path
    prompt = 'The Sun is yellow because'
    num_tokens = 128

    tokenizer = Tokenizer(model_path)
    tokenized_input = tokenizer.encode(prompt)
    # ignore_eos=True to ensure model will generate exactly num_tokens
    encoded_results = npu_model.generate(tokenized_input, max_new_tokens=num_tokens, ignore_eos=True)
    assert len(encoded_results.tokens[0]) == num_tokens


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("npu_config", PIPELINE_CONFIGS, indirect=True)
def test_terminate_by_out_of_memory(
    llm_model: OVConvertedModelSchema, 
    npu_config: dict
):
    model_path = llm_model.models_path
    prompt = 'The Sun is yellow because'
    pipeline_config = { "MAX_PROMPT_LEN": 256, "MIN_RESPONSE_LEN": 64 }
    pipeline_config |= npu_config
    kv_cache_size = pipeline_config['MAX_PROMPT_LEN'] + pipeline_config['MIN_RESPONSE_LEN']

    tokenizer = Tokenizer(model_path)
    tokenized_input = tokenizer.encode(prompt)
    input_len = tokenized_input.input_ids.get_shape()[1]

    pipe = LLMPipeline(model_path, "NPU", **pipeline_config)
    encoded_results = pipe.generate(tokenized_input, max_new_tokens=1000, ignore_eos=True)

    assert len(encoded_results.tokens[0]) == (kv_cache_size - input_len + 1)


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("npu_config", PIPELINE_CONFIGS, indirect=True)
def test_terminate_by_sampler(
    llm_model: OVConvertedModelSchema, 
    npu_model: LLMPipeline,
):
    model_path = llm_model.models_path
    prompt = 'The Sun is yellow because'

    current_iter = 0
    num_iters = 10

    class TestStreamer(StreamerBase):
        def __init__(self):
            StreamerBase.__init__(self)

        def write(self, token_id) -> StreamingStatus:
            nonlocal current_iter
            current_iter += 1
            return StreamingStatus.RUNNING if current_iter != num_iters else StreamingStatus.STOP
        def end(self):
            pass

    tokenizer = Tokenizer(model_path)
    tokenized_input = tokenizer.encode(prompt)

    encoded_results = npu_model.generate(
        tokenized_input, 
        max_new_tokens=1000, 
        ignore_eos=True, 
        streamer=TestStreamer(),
    )

    assert len(encoded_results.tokens[0]) == num_iters


# FIXME: Known problem, output differs from stateful pipeline starting from 3rd prompt!
@pytest.mark.skip(reason="JIRA-144780: Output differs from stateful pipeline")
@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("npu_config", PIPELINE_CONFIGS, indirect=True)
def test_chat_generation(
    ov_model: LLMPipeline,
    npu_model: LLMPipeline,
):
    def generate_with_chat_mode(pipe: LLMPipeline, questions: list[str]) -> list[str]:
        pipe.start_chat()
        answers = [ pipe.generate(question, max_new_tokens=50, do_sample=False) for question in questions ]
        pipe.finish_chat()
        return answers
    
    def generate_with_chat_history(pipe: LLMPipeline, questions: list[str]) -> ChatHistory:
        chat_history = ChatHistory()
        for question in questions:
            chat_history.append({"role": "user", "content": question})
            decoded_results = pipe.generate(chat_history, max_new_tokens=50, do_sample=False)
            chat_history.append({"role": "assistant", "content": decoded_results.texts[0]})
        return chat_history
    
    questions = [
        '1+1=',
        'What is the previous answer?',
        'Why is the Sun yellow?',
        'What was my first question?'
    ]

    answers_chat_mode_stateful = generate_with_chat_mode(ov_model, questions)
    answers_chat_mode_static = generate_with_chat_mode(npu_model, questions)
    assert answers_chat_mode_stateful == answers_chat_mode_static, \
        f"CPU output:\n{answers_chat_mode_stateful}\nNPU output:\n{answers_chat_mode_static}"

    chat_history_stateful = generate_with_chat_history(ov_model, questions)
    messages_stateful = chat_history_stateful.get_messages()
    chat_history_static = generate_with_chat_history(npu_model, questions)
    messages_static = chat_history_static.get_messages()
    assert messages_stateful == messages_static, \
        f"CPU output:\n{messages_stateful}\nNPU output:\n{messages_static}"

    answers_chat_history_static = [msg["content"] for msg in messages_static if msg["role"] == "assistant"]
    assert answers_chat_mode_static == answers_chat_history_static, (
        f"NPU chat mode output:\n{answers_chat_mode_static}\n"
        f"NPU chat history output:\n{answers_chat_history_static}"
    )


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("npu_config", PIPELINE_CONFIGS, indirect=True)
def test_readonly_input_tensor(npu_model: LLMPipeline):
    input_ids = np.array([[1, 4, 42]], dtype=np.int64)
    input_ids.flags.writeable = False

    attention_mask = np.array([[1, 1, 1]], dtype=np.int64)
    attention_mask.flags.writeable = False

    inputs_ov = TokenizedInputs(ov.Tensor(input_ids), ov.Tensor(attention_mask))
    npu_model.generate(inputs_ov, max_new_tokens=5)

    readonly_tensor = ov.Tensor(input_ids)
    npu_model.generate(readonly_tensor, max_new_tokens=5)
