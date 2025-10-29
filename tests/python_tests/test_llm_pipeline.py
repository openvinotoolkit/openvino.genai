# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import gc
import sys
import os
import json
import numpy as np
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field

import openvino as ov
import openvino_genai as ov_genai

from utils.constants import get_default_llm_properties, extra_generate_kwargs
from utils.hugging_face import generation_config_to_hf, download_and_convert_model
# model_tmp_path fixture import required
from utils.tokenizers import delete_rt_info, model_tmp_path
from utils.ov_genai_pipelines import create_ov_pipeline, generate_and_compare, get_main_pipeline_types, PipelineType, GenerationChatInputsType
from data.models import get_models_list, get_chat_models_list


def assert_hf_equals_genai(hf_reference, genai_output):
    __tracebackhide__ = True
    assert hf_reference == genai_output, f"HF reference:\n{hf_reference}\nGenAI output:\n{genai_output}"


#
# e2e work
#

test_cases = [
    (dict(max_new_tokens=20), '你好！ 你好嗎？'),
    (dict(max_new_tokens=30, num_beams=15, num_beam_groups=3, num_return_sequences=15, diversity_penalty=1.0), 'Why is the Sun yellow?'),
]
@pytest.mark.parametrize("generation_config_dict,prompt", test_cases)
@pytest.mark.parametrize("model_id", get_models_list())
@pytest.mark.parametrize("pipeline_type", get_main_pipeline_types())
@pytest.mark.precommit
def test_string_inputs(model_id, generation_config_dict, prompt, pipeline_type):
    generate_and_compare(model=model_id, prompts=[prompt], generation_config=generation_config_dict, pipeline_type=pipeline_type)


input_tensors_list = [
    # input_ids, attention_mask
    (np.array([[1, 4, 42]], dtype=np.int64), None),
    (np.array([[1, 4, 42]], dtype=np.int64), np.array([[1, 1, 1]], dtype=np.int64)),
]
@pytest.mark.parametrize("inputs", input_tensors_list)
@pytest.mark.parametrize("model_id", get_models_list())
@pytest.mark.precommit
def test_encoded_inputs(model_id, inputs):
    opt_model, hf_tokenizer, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)

    ov_generation_config = ov_genai.GenerationConfig(max_new_tokens=20)
    hf_generation_config = generation_config_to_hf(opt_model.generation_config, ov_generation_config)

    input_ids, attention_mask = inputs
    prompt_len = input_ids.shape[1]

    if attention_mask is not None:
        inputs_ov = ov_genai.TokenizedInputs(ov.Tensor(input_ids), ov.Tensor(attention_mask))
        inputs_hf = dict(inputs=torch.tensor(input_ids), attention_mask=torch.tensor(attention_mask))
    else:
        inputs_hf = dict(inputs=torch.tensor(input_ids))
        inputs_ov = ov.Tensor(input_ids)

    hf_output = opt_model.generate(**inputs_hf, generation_config=hf_generation_config, **extra_generate_kwargs()).sequences[0]
    ov_output = ov_pipe.generate(inputs_ov, ov_generation_config)

    hf_res = hf_output[prompt_len:].numpy()
    ov_res = np.array(ov_output.tokens, dtype=np.int64)
    assert np.all(ov_res == hf_res)


test_configs = [
    dict(max_new_tokens=20),
    dict(max_new_tokens=20, num_beam_groups=2, num_beams=6, diversity_penalty=1.0)
]
batched_prompts = [
    ['table is made', 'They sky is blue because', 'Difference between Jupiter and Mars is that'],
    ['hello', 'Here is the longest nowel ever: '],
    ['Alan Turing was a', 'return 0', '你好！ 你好嗎？'],
    ['table is made', 'table is made [force left pad tokens]']
]
@pytest.mark.parametrize("generation_config_dict", test_configs)
@pytest.mark.parametrize("prompts", batched_prompts)
@pytest.mark.parametrize("model_id", get_models_list())
@pytest.mark.parametrize("pipeline_type", get_main_pipeline_types())
@pytest.mark.precommit
def test_batch_string_inputs(model_id, generation_config_dict, prompts, pipeline_type):
    generate_and_compare(model=model_id, prompts=prompts, generation_config=generation_config_dict, pipeline_type=pipeline_type)


@pytest.mark.precommit
def test_batch_size_switch():
    model_id = 'katuni4ka/tiny-random-phi3'
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)

    ov_pipe.generate(["a"], max_new_tokens=2)
    ov_pipe.generate(["1", "2"], max_new_tokens=2)
    ov_pipe.generate(["a"], max_new_tokens=2)


@pytest.mark.precommit
def test_empty_encoded_inputs_throw():
    model_id = 'katuni4ka/tiny-random-phi3'
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)

    with pytest.raises(RuntimeError):
        ov_pipe.generate(ov.Tensor(np.array([[]], dtype=np.int64)), max_new_tokens=2)


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_chat_models_list())
def test_different_input_types_works_same_and_change_nothing(model_id):
    opt_model, hf_tokenizer, models_path  = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)

    ov_generation_config = ov_genai.GenerationConfig()
    ov_generation_config.max_new_tokens = 30
    ov_generation_config.apply_chat_template = False

    res_string_input_1 = ov_pipe.generate(questions[0], generation_config=ov_generation_config)

    tokenizer = ov_pipe.get_tokenizer()
    ov_tokens = tokenizer.encode(questions[0], add_special_tokens=True)
    res_encoded_input = ov_pipe.generate(ov_tokens, generation_config=ov_generation_config)
    res_encoded_input_str = hf_tokenizer.decode(res_encoded_input.tokens[0], skip_special_tokens=True)

    assert res_string_input_1 == res_encoded_input_str

    res_string_input_2 = ov_pipe.generate(questions[0], generation_config=ov_generation_config)

    assert res_string_input_1 == res_string_input_2

#
# Chat scenario
#

chat_inputs = [
    (dict(max_new_tokens=20), ""),
    (dict(max_new_tokens=20), "Pretend that 1+1=1"),
    (dict(max_new_tokens=10, num_beam_groups=3, num_beams=15, num_return_sequences=1, diversity_penalty=1.0), "")
]

questions = [
    '1+1=',
    'What is the previous answer?',
    'Why is the Sun yellow?',
    'What was my first question?'
]

@pytest.mark.parametrize("inputs", chat_inputs)
@pytest.mark.parametrize("model_id", get_chat_models_list())
@pytest.mark.parametrize("input_type", [
    GenerationChatInputsType.STRING,
    GenerationChatInputsType.ENCODED_INPUTS,
    GenerationChatInputsType.CHAT_HISTORY])
@pytest.mark.precommit
def test_chat_scenario(model_id, inputs, input_type):
    chat_history_hf = []
    chat_history_ov = ov_genai.ChatHistory() if input_type == GenerationChatInputsType.CHAT_HISTORY else []

    opt_model, hf_tokenizer, models_path = download_and_convert_model(model_id)
    if input_type == GenerationChatInputsType.ENCODED_INPUTS:
        # chat is not supported for PA backend with encoded_inputs format
        ov_pipe = create_ov_pipeline(models_path, pipeline_type=PipelineType.STATEFUL)
    else:
        ov_pipe = create_ov_pipeline(models_path)

    generation_config_kwargs, system_message = inputs

    ov_generation_config = ov_genai.GenerationConfig(**generation_config_kwargs)
    hf_generation_config = generation_config_to_hf(opt_model.generation_config, ov_generation_config)

    prev_chat_len = 0

    ov_pipe.start_chat(system_message)
    chat_history_hf.append({"role": "system", "content": system_message})
    chat_history_ov.append({"role": "system", "content": system_message})
    
    for prompt in questions:
        chat_history_hf.append({'role': 'user', 'content': prompt})
        chat_history_ov.append({'role': 'user', 'content': prompt})

        chat_prompt = hf_tokenizer.apply_chat_template(chat_history_hf, tokenize=False, add_generation_prompt=True)
        tokenized = hf_tokenizer(chat_prompt, return_tensors='pt', add_special_tokens=False)
        prompt_len = tokenized['input_ids'].numel()

        answer = opt_model.generate(**tokenized, generation_config=hf_generation_config, **extra_generate_kwargs()).sequences[0]
        answer_str = hf_tokenizer.decode(answer[prompt_len:], skip_special_tokens=True)
        chat_history_hf.append({'role': 'assistant', 'content': answer_str})

        if input_type == GenerationChatInputsType.STRING:
            answer_ov = ov_pipe.generate(prompt, generation_config=ov_generation_config)
        elif input_type == GenerationChatInputsType.CHAT_HISTORY:
            result_ov: ov_genai.DecodedResults = ov_pipe.generate(chat_history_ov, generation_config=ov_generation_config)
            answer_ov = result_ov.texts[0]
        elif input_type == GenerationChatInputsType.ENCODED_INPUTS:
            input_ids = np.array([tokenized['input_ids'][0][prev_chat_len:]], dtype=np.int64)
            attention_mask = np.array([tokenized['attention_mask'][0][prev_chat_len:]], dtype=np.int64)
            inputs_ov = ov_genai.TokenizedInputs(ov.Tensor(input_ids), ov.Tensor(attention_mask))

            result_ov = ov_pipe.generate(inputs_ov, generation_config=ov_generation_config).tokens[0]

            answer_ov = hf_tokenizer.decode(result_ov, skip_special_tokens=True)
            prev_chat_len = len(tokenized['input_ids'][0]) + len(result_ov)

        chat_history_ov.append({'role': 'assistant', 'content': answer_ov})

    ov_pipe.finish_chat()

    chat_history_messages_ov = chat_history_ov.get_messages() if input_type == GenerationChatInputsType.CHAT_HISTORY else chat_history_ov
    assert_hf_equals_genai(chat_history_hf, chat_history_messages_ov)

    # Test chat history generate without start_chat/finish_chat matches the same chat scenario
    if input_type == GenerationChatInputsType.CHAT_HISTORY:
        chat_history_ov = ov_genai.ChatHistory()
        chat_history_ov.append({"role": "system", "content": system_message})

        for prompt in questions:
            chat_history_ov.append({"role": "user", "content": prompt})
            result_ov: ov_genai.DecodedResults = ov_pipe.generate(chat_history_ov, generation_config=ov_generation_config)
            answer_ov = result_ov.texts[0]
            chat_history_ov.append({"role": "assistant", "content": answer_ov})

        chat_history_messages_ov = chat_history_ov.get_messages()
        assert_hf_equals_genai(chat_history_hf, chat_history_messages_ov)


@pytest.mark.precommit
def test_chat_scenario_several_chats_in_series():
    opt_model, hf_tokenizer, models_path  = download_and_convert_model(get_chat_models_list()[0])
    ov_pipe = create_ov_pipeline(models_path)

    generation_config_kwargs, _ = chat_inputs[0]
    ov_generation_config = ov_genai.GenerationConfig(**generation_config_kwargs)
    hf_generation_config = generation_config_to_hf(opt_model.generation_config, ov_generation_config)

    for i in range(2):
        chat_history_hf = []
        chat_history_ov = []
        ov_pipe.start_chat()
        for prompt in questions[:2]:
            chat_history_hf.append({'role': 'user', 'content': prompt})
            chat_history_ov.append({'role': 'user', 'content': prompt})

            chat_prompt = hf_tokenizer.apply_chat_template(chat_history_hf, tokenize=False, add_generation_prompt=True)
            tokenized = hf_tokenizer(chat_prompt, return_tensors='pt', add_special_tokens=False)
            prompt_len = tokenized['input_ids'].numel()
    
            answer = opt_model.generate(**tokenized, generation_config=hf_generation_config, **extra_generate_kwargs()).sequences[0]
            answer_str = hf_tokenizer.decode(answer[prompt_len:], skip_special_tokens=True)
            chat_history_hf.append({'role': 'assistant', 'content': answer_str})
    
            answer_ov = ov_pipe.generate(prompt, generation_config=ov_generation_config)
            chat_history_ov.append({'role': 'assistant', 'content': answer_ov})

        ov_pipe.finish_chat()

        assert_hf_equals_genai(chat_history_hf, chat_history_ov)


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_chat_models_list())
def test_chat_scenario_several_start(model_id):
    opt_model, hf_tokenizer, models_path  = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)

    generation_config_kwargs, _ = chat_inputs[0]
    ov_generation_config = ov_genai.GenerationConfig(**generation_config_kwargs)

    ov_pipe.start_chat()
    ov_pipe.start_chat()
    ov_pipe.generate(questions[0], generation_config=ov_generation_config)
    ov_pipe.finish_chat()


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_chat_models_list())
def test_generate_works_same_before_and_after_chat(model_id):
    opt_model, hf_tokenizer, models_path  = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)

    generation_config_kwargs, _ = chat_inputs[0]
    ov_generation_config = ov_genai.GenerationConfig(**generation_config_kwargs)
    ov_generation_config.apply_chat_template = False

    res_before_chat = ov_pipe.generate(questions[0], generation_config=ov_generation_config)

    ov_pipe.start_chat()
    ov_pipe.generate(questions[0], generation_config=ov_generation_config)
    ov_pipe.finish_chat()

    res_after_chat = ov_pipe.generate(questions[0], generation_config=ov_generation_config)
    
    assert res_after_chat == res_before_chat

#
# Streaming with callback
#

def user_defined_callback(subword):
    print(subword)


def user_defined_status_callback(subword):
    print(subword)
    return ov_genai.StreamingStatus.RUNNING


@pytest.mark.parametrize("callback", [print, user_defined_callback, user_defined_status_callback, lambda subword: print(subword)])
@pytest.mark.parametrize("model_id", get_models_list())
@pytest.mark.precommit
def test_callback_one_string(callback, model_id):
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 10
    ov_pipe.generate('table is made of', generation_config, callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, user_defined_status_callback, lambda subword: print(subword)])
@pytest.mark.parametrize("model_id", get_models_list())
@pytest.mark.precommit
def test_callback_batch_throws(callback, model_id):
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    with pytest.raises(RuntimeError):
        ov_pipe.generate(['1', '2'], ov_pipe.get_generation_config(), callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, user_defined_status_callback, lambda subword: print(subword)])
@pytest.mark.parametrize("model_id", get_models_list())
@pytest.mark.precommit
def test_callback_kwargs_one_string(callback, model_id):
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    ov_pipe.generate('table is made of', max_new_tokens=10, streamer=callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, user_defined_status_callback, lambda subword: print(subword)])
@pytest.mark.parametrize("model_id", get_models_list())
@pytest.mark.precommit
def test_callback_decoding_metallama(model_id, callback):
    # On metallama this prompt generates output which can shorten after adding new tokens.
    # Test that streamer correctly handles such cases.
    prompt = 'I have an interview about product speccing with the company Weekend Health. Give me an example of a question they might ask with regards about a new feature'
    if model_id != 'meta-llama/Meta-Llama-3-8B-Instruct':
        pytest.skip()
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    ov_pipe.generate(prompt, max_new_tokens=300, streamer=callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, user_defined_status_callback, lambda subword: print(subword)])
@pytest.mark.parametrize("model_id", get_models_list())
@pytest.mark.precommit
def test_callback_kwargs_batch_throws(callback, model_id):
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    with pytest.raises(RuntimeError):
        ov_pipe.generate(['1', '2'], max_new_tokens=10, streamer=callback)


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_models_list())
def test_callback_terminate_by_bool(model_id):
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)

    current_iter = 0
    num_iters = 10
    def callback(subword):
        nonlocal current_iter
        current_iter += 1
        return current_iter == num_iters

    max_new_tokens = 100
    ov_generation_config = ov_genai.GenerationConfig(max_new_tokens=max_new_tokens, ignore_eos=True)

    # without attention mask
    input_ids, _ = input_tensors_list[0]
    inputs_ov = ov.Tensor(input_ids)
    ov_output = ov_pipe.generate(inputs_ov, ov_generation_config, streamer=callback)

    assert len(ov_output.tokens[0]) < max_new_tokens


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_models_list())
def test_callback_terminate_by_status(model_id):
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)

    current_iter = 0
    num_iters = 10
    def callback(subword):
        nonlocal current_iter
        current_iter += 1
        return ov_genai.StreamingStatus.STOP if current_iter == num_iters else ov_genai.StreamingStatus.RUNNING

    max_new_tokens = 100
    ov_generation_config = ov_genai.GenerationConfig(max_new_tokens=max_new_tokens, ignore_eos=True)

    # without attention mask
    input_ids, _ = input_tensors_list[0]
    inputs_ov = ov.Tensor(input_ids)
    ov_output = ov_pipe.generate(inputs_ov, ov_generation_config, streamer=callback)

    assert len(ov_output.tokens[0]) < max_new_tokens


@pytest.mark.parametrize("model_id", get_chat_models_list())
@pytest.mark.precommit
def test_chat_scenario_callback_cancel(model_id):
    callback_questions = [
        '1+1=',
        'Why is the Sun yellow?',
        'What is the previous answer?',
        'What was my first question?'
    ]

    generation_config_kwargs = dict(max_new_tokens=20)

    chat_history_hf = []
    chat_history_ov = []

    opt_model, hf_tokenizer, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)

    ov_generation_config = ov_genai.GenerationConfig(**generation_config_kwargs)
    hf_generation_config = generation_config_to_hf(opt_model.generation_config, ov_generation_config)
    
    current_iter = 0
    num_iters = 3
    def callback(subword):
        nonlocal current_iter
        current_iter += 1
        return ov_genai.StreamingStatus.CANCEL if current_iter == num_iters else ov_genai.StreamingStatus.RUNNING

    ov_pipe.start_chat()
    for prompt in callback_questions:
        if (prompt != callback_questions[1]):
            chat_history_hf.append({'role': 'user', 'content': prompt})
            chat_history_ov.append({'role': 'user', 'content': prompt})

            chat_prompt = hf_tokenizer.apply_chat_template(chat_history_hf, tokenize=False, add_generation_prompt=True)
            tokenized = hf_tokenizer(chat_prompt, return_tensors='pt', add_special_tokens=False)
            prompt_len = tokenized['input_ids'].numel()

            answer = opt_model.generate(**tokenized, generation_config=hf_generation_config, **extra_generate_kwargs()).sequences[0]
            answer_str = hf_tokenizer.decode(answer[prompt_len:], skip_special_tokens=True)
            chat_history_hf.append({'role': 'assistant', 'content': answer_str})

            answer_ov = ov_pipe.generate(prompt, generation_config=ov_generation_config)
            chat_history_ov.append({'role': 'assistant', 'content': answer_ov})
        else:
            answer_ov = ov_pipe.generate(prompt, generation_config=ov_generation_config, streamer=callback)

    ov_pipe.finish_chat()

    assert_hf_equals_genai(chat_history_hf, chat_history_ov)


class PrinterNone(ov_genai.StreamerBase):
    def __init__(self, tokenizer):
        # super() may work, but once you begin mixing Python and C++
        # multiple inheritance, things will fall apart due to
        # differences between Python’s MRO and C++’s mechanisms.
        ov_genai.StreamerBase.__init__(self)
        self.tokenizer = tokenizer
    def put(self, token_id):
        # print(self.tokenizer.decode([token_id]))  # Incorrect way to print, but easy to implement
        print(token_id)  # print only token because self.tokenizer.decode([token_id]) are not implemented yet
    def end(self):
        print('end')


class PrinterBool(ov_genai.StreamerBase):
    def __init__(self, tokenizer):
        # super() may work, but once you begin mixing Python and C++
        # multiple inheritance, things will fall apart due to
        # differences between Python’s MRO and C++’s mechanisms.
        ov_genai.StreamerBase.__init__(self)
        self.tokenizer = tokenizer
    def put(self, token_id):
        # print(self.tokenizer.decode([token_id]))  # Incorrect way to print, but easy to implement
        print(token_id)  # print only token because self.tokenizer.decode([token_id]) are not implemented yet
        return False
    def end(self):
        print('end')


class PrinterStatus(ov_genai.StreamerBase):
    def __init__(self, tokenizer):
        # super() may work, but once you begin mixing Python and C++
        # multiple inheritance, things will fall apart due to
        # differences between Python’s MRO and C++’s mechanisms.
        ov_genai.StreamerBase.__init__(self)
        self.tokenizer = tokenizer
    def write(self, token_id):
        # print(self.tokenizer.decode([token_id]))  # Incorrect way to print, but easy to implement
        print(token_id)  # print only token because self.tokenizer.decode([token_id]) are not implemented yet
        return ov_genai.StreamingStatus.RUNNING
    def end(self):
        print('end')


@pytest.mark.parametrize("streamer_base", [PrinterNone, PrinterBool, PrinterStatus])
@pytest.mark.parametrize("model_id", get_models_list())
@pytest.mark.precommit
def test_streamer_one_string(streamer_base, model_id):
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 10
    printer = streamer_base(ov_pipe.get_tokenizer())
    ov_pipe.generate('table is made of', generation_config, printer)


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_models_list())
def test_streamer_batch_throws(model_id):
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    printer = PrinterNone(ov_pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        ov_pipe.generate(['1', '2'], ov_pipe.get_generation_config(), printer)


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_models_list())
def test_streamer_kwargs_one_string(model_id):
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    printer = PrinterNone(ov_pipe.get_tokenizer())
    ov_pipe.generate('table is made of', max_new_tokens=10, do_sample=False, streamer=printer)


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_models_list())
def test_streamer_kwargs_batch_throws(model_id):
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    printer = PrinterNone(ov_pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        ov_pipe.generate('', num_beams=2, streamer=printer)


@pytest.mark.precommit
@pytest.mark.parametrize("callback", [print, user_defined_callback, user_defined_status_callback, lambda subword: print(subword)])
@pytest.mark.parametrize("model_id", get_models_list())
def test_operator_with_callback_one_string(callback, model_id):
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    ten_tokens = ov_pipe.get_generation_config()
    ten_tokens.max_new_tokens = 10
    ov_pipe('talbe is made of', ten_tokens, callback)


@pytest.mark.precommit
@pytest.mark.parametrize("callback", [print, user_defined_callback, user_defined_status_callback, lambda subword: print(subword)])
@pytest.mark.parametrize("model_id", get_models_list())
def test_operator_with_callback_batch_throws(callback, model_id):
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    with pytest.raises(RuntimeError):
        ov_pipe(['1', '2'], ov_pipe.get_generation_config(), callback)


@pytest.mark.parametrize("streamer_base", [PrinterNone, PrinterBool, PrinterStatus])
@pytest.mark.parametrize("model_id", get_models_list())
@pytest.mark.precommit
def test_operator_with_streamer_kwargs_one_string(streamer_base, model_id):
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    printer = streamer_base(ov_pipe.get_tokenizer())
    ov_pipe('hi', max_new_tokens=10, do_sample=True, streamer=printer)


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_models_list())
def test_operator_with_streamer_kwargs_batch_throws(model_id):
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    printer = PrinterNone(ov_pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        ov_pipe('', num_beams=2, streamer=printer)

#
# Tests on generation configs handling
#


def load_genai_pipe_with_configs(configs: list[tuple], temp_path):
    # Load LLMPipeline where all configs are cleared.
    # remove existing jsons from previous tests
    for json_file in temp_path.glob("*.json"):
        json_file.unlink()
    delete_rt_info(configs, temp_path)

    for config_json, config_name in configs:
        with (temp_path / config_name).open('w', encoding="utf-8") as f:
            json.dump(config_json, f)

    ov_pipe = ov_genai.LLMPipeline(temp_path, 'CPU', **get_default_llm_properties())

    for _, config_name in configs:
        os.remove(temp_path / config_name)

    return ov_pipe


@pytest.mark.precommit
def test_eos_token_is_inherited_from_default_generation_config(model_tmp_path):
    _, temp_path = model_tmp_path
    ov_pipe = load_genai_pipe_with_configs([({"eos_token_id": 37}, "config.json")], temp_path)

    config = ov_genai.GenerationConfig()
    config.do_sample = True  # no eos_token_id but it's loaded from config.json
    ov_pipe.set_generation_config(config)

    assert 37 == ov_pipe.get_generation_config().eos_token_id


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_models_list())
def test_pipeline_validates_generation_config(model_id):
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    invalid_generation_config = dict(num_beam_groups=3, num_beams=15, do_sample=True) # beam sample is not supported
    with pytest.raises(RuntimeError):
        ov_pipe.generate("dummy prompt", **invalid_generation_config)

#
# Work with Unicode in Python API
#

@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_models_list())
def test_unicode_pybind_decoding_one_string(model_id):
    # On this model this prompt generates unfinished utf string.
    # Test that pybind will not fail.
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    res_str = ov_pipe.generate(',', max_new_tokens=4, apply_chat_template=False)
    assert '�' == res_str[-1]


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_models_list())
def test_unicode_pybind_decoding_batched(model_id):
    # On this model this prompt generates unfinished utf string.
    # Test that pybind will not fail.
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    res_str = ov_pipe.generate([","], max_new_tokens=4, apply_chat_template=False)
    assert '�' == res_str.texts[0][-1]


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_models_list())
def test_unicode_pybind_decoding_one_string_streamer(model_id):
    # On this model this prompt generates unfinished utf-8 string
    # and streams it. Test that pybind will not fail while we pass string to python.
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    res_str = []
    ov_pipe.generate(",", max_new_tokens=4, apply_chat_template=False, streamer=lambda x: res_str.append(x))
    assert '�' == ''.join(res_str)[-1]

#
# Perf metrics
#

def run_perf_metrics_collection(model_id, generation_config_dict: dict, prompt: str) -> ov_genai.PerfMetrics:
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    return ov_pipe.generate([prompt], **generation_config_dict).perf_metrics


test_cases = [
    (dict(max_new_tokens=20), 'table is made of'),
]
@pytest.mark.parametrize("generation_config,prompt", test_cases)
@pytest.mark.precommit
def test_perf_metrics(generation_config, prompt):
    import time
    start_time = time.perf_counter()
    model_id = 'katuni4ka/tiny-random-gemma2'
    perf_metrics = run_perf_metrics_collection(model_id, generation_config, prompt)
    total_time = (time.perf_counter() - start_time) * 1000

    # Check that load time is adequate.
    load_time = perf_metrics.get_load_time()
    assert load_time > 0 and load_time < total_time

    # Check that num input and generated tokens are adequate.
    num_generated_tokens = perf_metrics.get_num_generated_tokens()
    assert num_generated_tokens > 0 and num_generated_tokens <= generation_config['max_new_tokens']

    num_input_tokens = perf_metrics.get_num_input_tokens()
    assert num_input_tokens > 0 and num_input_tokens <= len(prompt)

    mean_ttft, std_ttft = perf_metrics.get_ttft()
    assert (mean_ttft, std_ttft) == (perf_metrics.get_ttft().mean, perf_metrics.get_ttft().std)
    assert mean_ttft > 0 and mean_ttft < 1000.0

    raw_metrics = perf_metrics.raw_metrics
    durations = np.array(raw_metrics.m_durations) / 1000
    # Check that prefill is not included in durations for TPOT calculation.
    # For the very long prompt prefill is slow and TTFT is much larger than any other token generation duration.
    assert np.all(mean_ttft > durations)

    mean_tpot, std_tpot = perf_metrics.get_tpot()
    assert (mean_tpot, std_tpot) == (perf_metrics.get_tpot().mean, perf_metrics.get_tpot().std)
    assert mean_tpot > 0 and mean_ttft < 1000.0

    mean_throughput, std_throughput = perf_metrics.get_throughput()
    assert (mean_throughput, std_throughput) == (perf_metrics.get_throughput().mean, perf_metrics.get_throughput().std)
    assert mean_throughput > 0 and mean_throughput < 20000.0

    mean_gen_duration, std_gen_duration = perf_metrics.get_generate_duration()
    assert (mean_gen_duration, std_gen_duration) == (perf_metrics.get_generate_duration().mean, perf_metrics.get_generate_duration().std)
    assert mean_gen_duration > 0 and load_time + mean_gen_duration < total_time
    assert std_gen_duration == 0

    mean_tok_duration, std_tok_duration = perf_metrics.get_tokenization_duration()
    assert (mean_tok_duration, std_tok_duration) == (perf_metrics.get_tokenization_duration().mean, perf_metrics.get_tokenization_duration().std)
    assert mean_tok_duration > 0 and mean_tok_duration < mean_gen_duration
    assert std_tok_duration == 0

    mean_detok_duration, std_detok_duration = perf_metrics.get_detokenization_duration()
    assert (mean_detok_duration, std_detok_duration) == (perf_metrics.get_detokenization_duration().mean, perf_metrics.get_detokenization_duration().std)
    assert mean_detok_duration > 0 and mean_detok_duration < mean_gen_duration
    assert std_detok_duration == 0

    # assert that calculating statistics manually from the raw counters we get the same restults as from PerfMetrics
    assert np.allclose(mean_tpot, np.mean(durations))
    assert np.allclose(std_tpot, np.std(durations), atol=0.00002)

    raw_dur = np.array(raw_metrics.generate_durations) / 1000
    assert np.allclose(mean_gen_duration, np.mean(raw_dur))
    assert np.allclose(std_gen_duration, np.std(raw_dur))

    raw_dur = np.array(raw_metrics.tokenization_durations) / 1000
    assert np.allclose(mean_tok_duration, np.mean(raw_dur))
    assert np.allclose(std_tok_duration, np.std(raw_dur))

    raw_dur = np.array(raw_metrics.detokenization_durations) / 1000
    assert np.allclose(mean_detok_duration, np.mean(raw_dur))
    assert np.allclose(std_detok_duration, np.std(raw_dur))

    assert len(raw_metrics.m_times_to_first_token) > 0
    assert len(raw_metrics.m_batch_sizes) > 0
    assert len(raw_metrics.m_durations) > 0


test_cases = [
    (dict(max_new_tokens=20), 'Generate json of a person'),
]
@pytest.mark.parametrize("generation_config,prompt", test_cases)
@pytest.mark.precommit
def test_perf_metrics_with_structured_output(generation_config, prompt):
    class Person(BaseModel):
        name: str = Field(pattern=r"^[A-Z][a-z]{1,20}$")
        surname: str = Field(pattern=r"^[A-Z][a-z]{1,20}$")
        age: int
        city: Literal["Dublin", "Dubai", "Munich"]
    generation_config.update(dict(structured_output_config=ov_genai.StructuredOutputConfig(json_schema=json.dumps(Person.model_json_schema()))))
    
    model_id = 'katuni4ka/tiny-random-gemma2'
    _, _, models_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)
    perf_metrics = ov_pipe.generate([prompt], **generation_config).perf_metrics
    raw_metrics = perf_metrics.raw_metrics

    assert len(perf_metrics.get_grammar_compiler_init_times()) > 0
    assert 'xgrammar' in perf_metrics.get_grammar_compiler_init_times() and perf_metrics.get_grammar_compiler_init_times()['xgrammar'] > 0.0
    
    assert len(raw_metrics.grammar_compile_times) > 0

    raw_compile_times = np.array(raw_metrics.grammar_compile_times) / 1000
    assert np.allclose(np.mean(raw_compile_times), perf_metrics.get_grammar_compile_time().mean)
    assert np.allclose(np.std(raw_compile_times), perf_metrics.get_grammar_compile_time().std)
    assert np.allclose(np.min(raw_compile_times), perf_metrics.get_grammar_compile_time().min)
    assert np.allclose(np.max(raw_compile_times), perf_metrics.get_grammar_compile_time().max)

    # Check that metrics are correctly accumulated/concatenated
    perf_metrics_2 = ov_pipe.generate([prompt], **generation_config).perf_metrics
    raw_metrics_2 = perf_metrics_2.raw_metrics
    accumulated_metrics = perf_metrics + perf_metrics_2
    assert accumulated_metrics.raw_metrics.grammar_compile_times == raw_metrics.grammar_compile_times + raw_metrics_2.grammar_compile_times


@pytest.mark.parametrize("pipeline_type", get_main_pipeline_types())
@pytest.mark.parametrize("stop_str", {True, False})
@pytest.mark.precommit
def test_pipelines_generate_with_streaming(pipeline_type, stop_str):
    # streamer
    it_cnt = 0
    def py_streamer(py_str: str):
        nonlocal it_cnt
        it_cnt += 1
        return False
    
    prompt = "Prompt example is"
    model_id : str = "facebook/opt-125m"

    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = 10
    if stop_str:    
        generation_config.stop_strings = {" the", "Prom"}
        generation_config.include_stop_str_in_output = False

    _ = generate_and_compare(model=model_id,
                             prompts=prompt,
                             generation_config=generation_config,
                             pipeline_type=pipeline_type,
                             streamer=py_streamer)
    if stop_str:
        assert it_cnt == 0
    else:
        assert it_cnt > 0

