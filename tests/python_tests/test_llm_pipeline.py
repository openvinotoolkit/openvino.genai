# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_genai as ov_genai
from openvino_genai import GenerationConfig
import pytest
from typing import Union, List, Dict, Optional
import numpy as np
import openvino as ov
import sys
from pathlib import Path
import torch

from common import run_llm_pipeline_with_ref
from ov_genai_test_utils import (
    get_models_list,
    read_model,
    load_genai_pipe_with_configs,
    get_chat_models_list,
    model_tmp_path,
)
from utils.hugging_face import generation_config_to_hf
#
# e2e work
#

test_cases = [
    (dict(max_new_tokens=20), '你好！ 你好嗎？'),
    (dict(max_new_tokens=30, num_beams=15, num_beam_groups=3, num_return_sequences=15, diversity_penalty=1.0), 'Why is the Sun yellow?'),
]
@pytest.mark.parametrize("generation_config_dict,prompt", test_cases)
@pytest.mark.parametrize("model_descr", get_models_list())
@pytest.mark.precommit
@pytest.mark.nightly
def test_string_inputs(model_descr, generation_config_dict, prompt):
    run_llm_pipeline_with_ref(model_id=model_descr[0], prompts=[prompt], generation_config=generation_config_dict, tmp_path=model_descr[1])


input_tensors_list = [
    # input_ids, attention_mask
    (np.array([[1, 4, 42]], dtype=np.int64), None),
    (np.array([[1, 4, 42]], dtype=np.int64), np.array([[1, 1, 1]], dtype=np.int64)),
]
@pytest.mark.parametrize("inputs", input_tensors_list)
@pytest.mark.parametrize("model_descr", get_models_list())
@pytest.mark.precommit
@pytest.mark.nightly
def test_encoded_inputs(model_descr, inputs):
    device = 'CPU'
    model_id, path, hf_tokenizer, opt_model, ov_pipe = read_model(model_descr)

    ov_generation_config = GenerationConfig(max_new_tokens=20)
    hf_generation_config = generation_config_to_hf(opt_model.generation_config, ov_generation_config)

    input_ids, attention_mask = inputs
    prompt_len = input_ids.shape[1]

    if attention_mask is not None:
        inputs_ov = ov_genai.TokenizedInputs(ov.Tensor(input_ids), ov.Tensor(attention_mask))
        inputs_hf = dict(inputs=torch.tensor(input_ids), attention_mask=torch.tensor(attention_mask))
    else:
        inputs_hf = dict(inputs=torch.tensor(input_ids))
        inputs_ov = ov.Tensor(input_ids)

    hf_output = opt_model.generate(**inputs_hf, generation_config=hf_generation_config).sequences[0]
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
@pytest.mark.parametrize("model_descr", get_models_list())
@pytest.mark.precommit
@pytest.mark.nightly
def test_batch_string_inputs(model_descr, generation_config_dict, prompts):
    run_llm_pipeline_with_ref(model_id=model_descr[0], prompts=prompts, generation_config=generation_config_dict, tmp_path=model_descr[1])


@pytest.mark.precommit
@pytest.mark.nightly
def test_batch_size_switch():
    ov_pipe = read_model(('katuni4ka/tiny-random-phi3', Path('tiny-random-phi3')))[4]
    ov_pipe.generate(["a"], max_new_tokens=2)
    ov_pipe.generate(["1", "2"], max_new_tokens=2)
    ov_pipe.generate(["a"], max_new_tokens=2)


@pytest.mark.precommit
@pytest.mark.nightly
def test_empty_encoded_inputs_throw():
    ov_pipe = read_model(('katuni4ka/tiny-random-phi3', Path('tiny-random-phi3')))[4]
    with pytest.raises(RuntimeError):
        ov_pipe.generate(ov.Tensor(np.array([[]], dtype=np.int64)), max_new_tokens=2)

#
# Chat scenario
#

chat_intpus = [
    (dict(max_new_tokens=20),  ""),
    (dict(max_new_tokens=20),  "You are a helpful assistant."),
    (dict(max_new_tokens=10, num_beam_groups=3, num_beams=15, num_return_sequences=1, diversity_penalty=1.0), "")
]

questions = [
    '1+1=',
    'What is the previous answer?',
    'Why is the Sun yellow?',
    'What was my first question?'
]

@pytest.mark.parametrize("intpus", chat_intpus)
@pytest.mark.parametrize("model_descr", get_chat_models_list())
@pytest.mark.precommit
@pytest.mark.nightly
def test_chat_scenario(model_descr, intpus):
    chat_history_hf = []
    chat_history_ov = []

    model_id, path, tokenizer, opt_model, ov_pipe = read_model((model_descr[0], model_descr[1]))

    generation_config_kwargs, system_massage = intpus

    ov_generation_config = GenerationConfig(**generation_config_kwargs)
    hf_generation_config = generation_config_to_hf(opt_model.generation_config, ov_generation_config)

    ov_pipe.start_chat(system_massage)
    chat_history_hf.append({"role": "system", "content": system_massage})
    chat_history_ov.append({"role": "system", "content": system_massage})
    for prompt in questions:
        chat_history_hf.append({'role': 'user', 'content': prompt})
        chat_history_ov.append({'role': 'user', 'content': prompt})

        chat_prompt = tokenizer.apply_chat_template(chat_history_hf, tokenize=False, add_generation_prompt=True)
        tokenized = tokenizer(chat_prompt, return_tensors='pt', add_special_tokens=False)
        prompt_len = tokenized['input_ids'].numel()

        answer = opt_model.generate(**tokenized, generation_config=hf_generation_config).sequences[0]
        answer_str = tokenizer.decode(answer[prompt_len:], skip_special_tokens=True)
        chat_history_hf.append({'role': 'assistant', 'content': answer_str})

        answer_ov = ov_pipe.generate(prompt, generation_config=ov_generation_config)
        chat_history_ov.append({'role': 'assistant', 'content': answer_ov})

    ov_pipe.finish_chat()

    if chat_history_ov != chat_history_hf:
        print(f'hf_output: {chat_history_hf}')
        print(f'ov_output: {chat_history_ov}')

    assert chat_history_ov == chat_history_hf


@pytest.mark.precommit
@pytest.mark.nightly
def test_chat_scenario_several_chats_in_series():
    model_descr = get_chat_models_list()[0]
    model_id, path, tokenizer, opt_model, ov_pipe = read_model((model_descr[0], model_descr[1]))

    generation_config_kwargs, _ = chat_intpus[0]
    ov_generation_config = GenerationConfig(**generation_config_kwargs)
    hf_generation_config = generation_config_to_hf(opt_model.generation_config, ov_generation_config)

    for i in range(2):
        chat_history_hf = []
        chat_history_ov = []
        ov_pipe.start_chat()
        for prompt in questions[:2]:
            chat_history_hf.append({'role': 'user', 'content': prompt})
            chat_history_ov.append({'role': 'user', 'content': prompt})

            chat_prompt = tokenizer.apply_chat_template(chat_history_hf, tokenize=False, add_generation_prompt=True)
            tokenized = tokenizer(chat_prompt, return_tensors='pt', add_special_tokens=False)
            prompt_len = tokenized['input_ids'].numel()

            answer = opt_model.generate(**tokenized, generation_config=hf_generation_config).sequences[0]
            answer_str = tokenizer.decode(answer[prompt_len:], skip_special_tokens=True)
            chat_history_hf.append({'role': 'assistant', 'content': answer_str})

            answer_ov = ov_pipe.generate(prompt, generation_config=ov_generation_config)
            chat_history_ov.append({'role': 'assistant', 'content': answer_ov})

        ov_pipe.finish_chat()

        if chat_history_ov != chat_history_hf:
            print(f'hf_output: {chat_history_hf}')
            print(f'ov_output: {chat_history_ov}')

        assert chat_history_ov == chat_history_hf


@pytest.mark.precommit
@pytest.mark.nightly
def test_chat_scenario_several_start():
    ov_pipe = read_model(get_chat_models_list()[0])[4]

    generation_config_kwargs, _ = chat_intpus[0]
    ov_generation_config = GenerationConfig(**generation_config_kwargs)

    ov_pipe.start_chat()
    ov_pipe.start_chat()
    ov_pipe.generate(questions[0], generation_config=ov_generation_config)
    ov_pipe.finish_chat()

#
# Streaming with callback
#

def user_defined_callback(subword):
    print(subword)


def user_defined_status_callback(subword):
    print(subword)
    return ov_genai.StreamingStatus.RUNNING


@pytest.mark.parametrize("callback", [print, user_defined_callback, user_defined_status_callback, lambda subword: print(subword)])
@pytest.mark.precommit
@pytest.mark.nightly
def test_callback_one_string(callback):
    ov_pipe = read_model(get_models_list()[0])[4]
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 10
    ov_pipe.generate('table is made of', generation_config, callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, user_defined_status_callback, lambda subword: print(subword)])
@pytest.mark.precommit
@pytest.mark.nightly
def test_callback_batch_throws(callback):
    ov_pipe = read_model(get_models_list()[0])[4]
    with pytest.raises(RuntimeError):
        ov_pipe.generate(['1', '2'], ov_pipe.get_generation_config(), callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, user_defined_status_callback, lambda subword: print(subword)])
@pytest.mark.precommit
@pytest.mark.nightly
def test_callback_kwargs_one_string(callback):
    pipe = read_model(get_models_list()[0])[4]
    pipe.generate('table is made of', max_new_tokens=10, streamer=callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, user_defined_status_callback, lambda subword: print(subword)])
@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("model_descr", get_models_list())
def test_callback_decoding_metallama(model_descr, callback):
    # On metallama this prompt generates output which can shorten after adding new tokens.
    # Test that streamer correctly handles such cases.
    prompt = 'I have an interview about product speccing with the company Weekend Health. Give me an example of a question they might ask with regards about a new feature'
    if model_descr[0] != 'meta-llama/Meta-Llama-3-8B-Instruct':
        pytest.skip()
    ov_pipe = read_model(model_descr)[4]
    ov_pipe.generate(prompt, max_new_tokens=300, streamer=callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, user_defined_status_callback, lambda subword: print(subword)])
@pytest.mark.precommit
@pytest.mark.nightly
def test_callback_kwargs_batch_throws(callback):
    pipe = read_model(get_models_list()[0])[4]
    with pytest.raises(RuntimeError):
        pipe.generate(['1', '2'], max_new_tokens=10, streamer=callback)


@pytest.mark.precommit
@pytest.mark.nightly
def test_callback_terminate_by_bool():
    pipe = read_model(get_models_list()[0])[4]

    current_iter = 0
    num_iters = 10
    def callback(subword):
        nonlocal current_iter
        current_iter += 1
        return current_iter == num_iters

    max_new_tokens = 100
    ov_generation_config = GenerationConfig(max_new_tokens=max_new_tokens, ignore_eos=True)

    # without attention mask
    input_ids, _ = input_tensors_list[0]
    inputs_ov = ov.Tensor(input_ids)
    ov_output = pipe.generate(inputs_ov, ov_generation_config, streamer=callback)

    assert len(ov_output.tokens[0]) < max_new_tokens


@pytest.mark.precommit
@pytest.mark.nightly
def test_callback_terminate_by_status():
    pipe = read_model(get_models_list()[0])[4]

    current_iter = 0
    num_iters = 10
    def callback(subword):
        nonlocal current_iter
        current_iter += 1
        return ov_genai.StreamingStatus.STOP if current_iter == num_iters else ov_genai.StreamingStatus.RUNNING

    max_new_tokens = 100
    ov_generation_config = GenerationConfig(max_new_tokens=max_new_tokens, ignore_eos=True)

    # without attention mask
    input_ids, _ = input_tensors_list[0]
    inputs_ov = ov.Tensor(input_ids)
    ov_output = pipe.generate(inputs_ov, ov_generation_config, streamer=callback)

    assert len(ov_output.tokens[0]) < max_new_tokens


@pytest.mark.parametrize("model_descr", get_chat_models_list())
@pytest.mark.precommit
@pytest.mark.nightly
def test_chat_scenario_callback_cancel(model_descr):
    callback_questions = [
        '1+1=',
        'Why is the Sun yellow?',
        'What is the previous answer?',
        'What was my first question?'
    ]

    generation_config_kwargs = dict(max_new_tokens=20)

    chat_history_hf = []
    chat_history_ov = []

    model_id, path, tokenizer, opt_model, ov_pipe = read_model((model_descr[0], model_descr[1] / '_test_chat'))

    ov_generation_config = GenerationConfig(**generation_config_kwargs)
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

            chat_prompt = tokenizer.apply_chat_template(chat_history_hf, tokenize=False, add_generation_prompt=True)
            tokenized = tokenizer(chat_prompt, return_tensors='pt', add_special_tokens=False)
            prompt_len = tokenized['input_ids'].numel()

            answer = opt_model.generate(**tokenized, generation_config=hf_generation_config).sequences[0]
            answer_str = tokenizer.decode(answer[prompt_len:], skip_special_tokens=True)
            chat_history_hf.append({'role': 'assistant', 'content': answer_str})

            answer_ov = ov_pipe.generate(prompt, generation_config=ov_generation_config)
            chat_history_ov.append({'role': 'assistant', 'content': answer_ov})
        else:
            answer_ov = ov_pipe.generate(prompt, generation_config=ov_generation_config, streamer=callback)

    ov_pipe.finish_chat()

    if chat_history_ov != chat_history_hf:
        print(f'hf_output: {chat_history_hf}')
        print(f'ov_output: {chat_history_ov}')

    assert chat_history_ov == chat_history_hf


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
@pytest.mark.precommit
@pytest.mark.nightly
def test_streamer_one_string(streamer_base):
    ov_pipe = read_model(get_models_list()[0])[4]
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 10
    printer = streamer_base(ov_pipe.get_tokenizer())
    ov_pipe.generate('table is made of', generation_config, printer)


@pytest.mark.precommit
@pytest.mark.nightly
def test_streamer_batch_throws():
    ov_pipe = read_model(get_models_list()[0])[4]
    printer = PrinterNone(ov_pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        ov_pipe.generate(['1', '2'], ov_pipe.get_generation_config(), printer)


@pytest.mark.precommit
@pytest.mark.nightly
def test_streamer_kwargs_one_string():
    ov_pipe = read_model(get_models_list()[0])[4]
    printer = PrinterNone(ov_pipe.get_tokenizer())
    ov_pipe.generate('table is made of', max_new_tokens=10, do_sample=False, streamer=printer)


@pytest.mark.precommit
@pytest.mark.nightly
def test_streamer_kwargs_batch_throws():
    ov_pipe = read_model(get_models_list()[0])[4]
    printer = PrinterNone(ov_pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        ov_pipe.generate('', num_beams=2, streamer=printer)


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("callback", [print, user_defined_callback, user_defined_status_callback, lambda subword: print(subword)])
def test_operator_with_callback_one_string(callback):
    ov_pipe = read_model(get_models_list()[0])[4]
    ten_tokens = ov_pipe.get_generation_config()
    ten_tokens.max_new_tokens = 10
    ov_pipe('talbe is made of', ten_tokens, callback)


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("callback", [print, user_defined_callback, user_defined_status_callback, lambda subword: print(subword)])
def test_operator_with_callback_batch_throws(callback):
    ov_pipe = read_model(get_models_list()[0])[4]
    with pytest.raises(RuntimeError):
        ov_pipe(['1', '2'], ov_pipe.get_generation_config(), callback)


@pytest.mark.parametrize("streamer_base", [PrinterNone, PrinterBool, PrinterStatus])
@pytest.mark.precommit
@pytest.mark.nightly
def test_operator_with_streamer_kwargs_one_string(streamer_base):
    ov_pipe = read_model(get_models_list()[0])[4]
    printer = streamer_base(ov_pipe.get_tokenizer())
    ov_pipe('hi', max_new_tokens=10, do_sample=True, streamer=printer)


@pytest.mark.precommit
@pytest.mark.nightly
def test_operator_with_streamer_kwargs_batch_throws():
    ov_pipe = read_model(get_models_list()[0])[4]
    printer = PrinterNone(ov_pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        ov_pipe('', num_beams=2, streamer=printer)

#
# Tests on generation configs handling
#

@pytest.mark.precommit
@pytest.mark.nightly
def test_eos_token_is_inherited_from_default_generation_config(model_tmp_path):
    model_id, temp_path = model_tmp_path
    ov_pipe = load_genai_pipe_with_configs([({"eos_token_id": 37}, "config.json")], temp_path)

    config = ov_genai.GenerationConfig()
    config.do_sample = True  # no eos_token_id but it's loaded from config.json
    ov_pipe.set_generation_config(config)

    assert 37 == ov_pipe.get_generation_config().eos_token_id


@pytest.mark.precommit
@pytest.mark.nightly
def test_pipeline_validates_generation_config():
    model_id, path = 'katuni4ka/tiny-random-phi3', Path('tiny-random-phi3')
    ov_pipe = read_model((model_id, path))[4]
    invalid_generation_config = dict(num_beam_groups=3, num_beams=15, do_sample=True) # beam sample is not supported
    with pytest.raises(RuntimeError):
        ov_pipe.generate("dummy prompt", **invalid_generation_config)

#
# Work with Unicode in Python API
#

@pytest.mark.precommit
@pytest.mark.nightly
def test_unicode_pybind_decoding_one_string():
    # On this model this prompt generates unfinished utf string.
    # Test that pybind will not fail.
    model_id, path = 'katuni4ka/tiny-random-phi3', Path('tiny-random-phi3')
    ov_pipe = read_model((model_id, path))[4]
    res_str = ov_pipe.generate(',', max_new_tokens=4, apply_chat_template=False)
    assert '�' == res_str[-1]


@pytest.mark.precommit
@pytest.mark.nightly
def test_unicode_pybind_decoding_batched():
    # On this model this prompt generates unfinished utf string.
    # Test that pybind will not fail.
    model_id, path = 'katuni4ka/tiny-random-phi3', Path('tiny-random-phi3')
    ov_pipe = read_model((model_id, path))[4]
    res_str = ov_pipe.generate([","], max_new_tokens=4, apply_chat_template=False)
    assert '�' == res_str.texts[0][-1]


@pytest.mark.precommit
@pytest.mark.nightly
def test_unicode_pybind_decoding_one_string_streamer():
    # On this model this prompt generates unfinished utf-8 string
    # and streams it. Test that pybind will not fail while we pass string to python.
    model_id, path = 'katuni4ka/tiny-random-phi3', Path('tiny-random-phi3')
    ov_pipe = read_model((model_id, path))[4]
    res_str = []
    ov_pipe.generate(",", max_new_tokens=4, apply_chat_template=False, streamer=lambda x: res_str.append(x))
    assert '�' == ''.join(res_str)[-1]

#
# Perf metrics
#

def run_perf_metrics_collection(model_descr, generation_config_dict: dict, prompt: str) -> ov_genai.PerfMetrics:
    model_id, path, hf_tokenizer, opt_model, ov_pipe = read_model(model_descr)
    return ov_pipe.generate([prompt], **generation_config_dict).perf_metrics


test_cases = [
    (dict(max_new_tokens=20), 'table is made of'),
]
@pytest.mark.parametrize("generation_config,prompt", test_cases)
@pytest.mark.precommit
@pytest.mark.nightly
def test_perf_metrics(generation_config, prompt):
    import time
    start_time = time.perf_counter()
    model_id, path = 'katuni4ka/tiny-random-gemma2', Path('katuni4ka-tiny-random-gemma2')
    perf_metrics = run_perf_metrics_collection((model_id, path), generation_config, prompt)
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
    assert np.all(mean_ttft > durations * 2)

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
    assert np.allclose(std_tpot, np.std(durations))

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
