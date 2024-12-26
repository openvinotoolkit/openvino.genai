# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_genai as ov_genai
import pytest
from typing import Dict, Tuple

from ov_genai_test_utils import (
    get_chat_models_list,
    read_model,
    get_continuous_batching,
)


generation_configs = [
    dict(do_sample=False, max_new_tokens=20),
    dict(do_sample=False, num_beam_groups=3, num_beams=15, num_return_sequences=1, max_new_tokens=10, diversity_penalty=1.0)
]


questions = [
    '1+1=',
    'What is the previous answer?',
    'Why is the Sun yellow?',
    'What was my first question?'
]


@pytest.mark.parametrize("generation_config", generation_configs)
@pytest.mark.parametrize("model_descr", get_chat_models_list())
@pytest.mark.precommit
@pytest.mark.nightly
def test_chat_compare_with_HF(model_descr, generation_config: Dict):
    chat_history_hf = []
    chat_history_ov = []
    chat_prompt = ''

    # Will set add_special_tokens=False inside pipeline when start_chat() is called.
    model_id, path, tokenizer, model_opt, pipe = read_model((model_descr[0], model_descr[1] / '_test_chat'))

    pipe.start_chat()
    for prompt in questions:
        chat_history_hf.append({'role': 'user', 'content': prompt})
        chat_history_ov.append({'role': 'user', 'content': prompt})

        chat_prompt = tokenizer.apply_chat_template(chat_history_hf, tokenize=False, add_generation_prompt=True)
        tokenized = tokenizer(chat_prompt, return_tensors='pt', add_special_tokens=False)

        answer = model_opt.generate(**tokenized, **generation_config)
        answer_str = tokenizer.decode(answer[0, tokenized['input_ids'].numel():], skip_special_tokens=True)
        chat_history_hf.append({'role': 'assistant', 'content': answer_str})

        answer_ov = pipe.generate(prompt, **generation_config)
        chat_history_ov.append({'role': 'assistant', 'content': answer_ov})

    pipe.finish_chat()

    if chat_history_ov != chat_history_hf:
        print(f'hf_output: {chat_history_hf}')
        print(f'ov_output: {chat_history_ov}')

    assert chat_history_ov == chat_history_hf


@pytest.mark.parametrize("generation_config", generation_configs)
@pytest.mark.parametrize("model_descr", get_chat_models_list())
@pytest.mark.precommit
@pytest.mark.nightly
def test_chat_compare_text_history_with_HF(model_descr, generation_config: Dict):
    # compares with HF when history in ov_genai is save as a text
    chat_history_hf = []
    chat_history_ov = []
    chat_prompt = ''

    # HF in chat scenario does not add special tokens, but openvino tokenizer by default is converted with add_special_tokens=True.
    # Need to regenerate openvino_tokenizer/detokenizer.
    model_id, path, hf_tokenizer, model_opt, ov_pipe = read_model((model_descr[0], model_descr[1] / '_test_chat'), add_special_tokens=False)
    ov_tokenizer = ov_pipe.get_tokenizer()

    for prompt in questions:
        chat_history_hf.append({'role': 'user', 'content': prompt})
        chat_history_ov.append({'role': 'user', 'content': prompt})

        chat_prompt = hf_tokenizer.apply_chat_template(chat_history_hf, tokenize=False, add_generation_prompt=True)
        tokenized = hf_tokenizer(chat_prompt, return_tensors='pt', add_special_tokens=False)

        answer = model_opt.generate(**tokenized, **generation_config)
        answer_str = hf_tokenizer.decode(answer[0, tokenized['input_ids'].numel():], skip_special_tokens=True)
        chat_history_hf.append({'role': 'assistant', 'content': answer_str})

        chat_prompt = ov_tokenizer.apply_chat_template(chat_history_ov, add_generation_prompt=True)
        answer_ov = ov_pipe.generate(chat_prompt, **generation_config)
        chat_history_ov.append({'role': 'assistant', 'content': answer_ov})

    if chat_history_ov != chat_history_hf:
        print(f'hf_output: {chat_history_hf}')
        print(f'ov_output: {chat_history_ov}')

    assert chat_history_ov == chat_history_hf


@pytest.mark.parametrize("generation_config", generation_configs[1:])
@pytest.mark.parametrize("model_descr", get_chat_models_list())
@pytest.mark.precommit
def test_chat_continuous_batching_vs_stateful(model_descr, generation_config: Dict):
    model_id, path, hf_tokenizer, opt_model, ov_stateful_pipe = read_model((model_descr[0], model_descr[1] / '_test_chat'))
    cb_pipe = get_continuous_batching(path)

    ov_stateful_pipe.start_chat()
    cb_pipe.start_chat()

    for question in questions:
        generated = cb_pipe.generate(question, **generation_config)
        reference = ov_stateful_pipe.generate(question, **generation_config)
        assert generated == reference

    # Test that finish_chat() doesn't fail just in case.
    cb_pipe.finish_chat()
