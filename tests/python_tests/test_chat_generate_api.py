# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_genai as ov_genai
import pytest
from typing import Dict, Tuple
from ov_genai_test_utils import (
    get_models_list,
    get_chat_models_list,
    read_model,
    load_tok,
    model_tmp_path,
    get_chat_templates,
    get_continuous_batching,
)


configs = [
    dict(do_sample=False, max_new_tokens=20),
    dict(do_sample=False, num_beam_groups=3, num_beams=15, num_return_sequences=1, max_new_tokens=10, diversity_penalty=1.0)
]


quenstions = [
    '1+1=',
    'What is the previous answer?',
    'Why is the Sun yellow?',
    'What was my first question?'
]


@pytest.mark.parametrize("generation_config", configs)
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
    for prompt in quenstions:
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


@pytest.mark.parametrize("generation_config", configs)
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
    model_id, path, tokenizer, model_opt, pipe = read_model((model_descr[0], model_descr[1] / '_test_chat'), add_special_tokens=False)
    
    for prompt in quenstions:
        chat_history_hf.append({'role': 'user', 'content': prompt})
        chat_history_ov.append({'role': 'user', 'content': prompt})
        
        chat_prompt = tokenizer.apply_chat_template(chat_history_hf, tokenize=False, add_generation_prompt=True)
        tokenized = tokenizer(chat_prompt, return_tensors='pt', add_special_tokens=False)
        
        answer = model_opt.generate(**tokenized, **generation_config)
        answer_str = tokenizer.decode(answer[0, tokenized['input_ids'].numel():], skip_special_tokens=True)
        chat_history_hf.append({'role': 'assistant', 'content': answer_str})
        
        chat_prompt = pipe.get_tokenizer().apply_chat_template(chat_history_ov, add_generation_prompt=True)
        answer_ov = pipe.generate(chat_prompt, **generation_config)
        chat_history_ov.append({'role': 'assistant', 'content': answer_ov})
  
    if chat_history_ov != chat_history_hf:
        print(f'hf_output: {chat_history_hf}')
        print(f'ov_output: {chat_history_ov}')
    assert chat_history_ov == chat_history_hf


@pytest.mark.parametrize("generation_config", configs)
@pytest.mark.parametrize("model_descr", get_chat_models_list())
@pytest.mark.precommit
@pytest.mark.nightly
def test_chat_compare_statefull_vs_text_history(model_descr, generation_config: Dict):
    # Check that when history is stored in KV cache results are the same as when history stored in a text.
    device ='CPU'
    
    chat_history_with_kv_cache = []
    chat_history_ov = []
    
    # HF in chat scenario does not add special tokens, but openvino tokenizer by default is converted with add_special_tokens=True.
    # Need to regenerate openvino_tokenizer/detokenizer.
    model_id, path, tokenizer, model_opt, pipe = read_model((model_descr[0], model_descr[1] / '_test_chat'), add_special_tokens=False)
    pipe_with_kv_cache = ov_genai.LLMPipeline(path, device, **{"ENABLE_MMAP": False})
  
    pipe_with_kv_cache.start_chat()
    for question in quenstions:
        chat_history_with_kv_cache.append({'role': 'user', 'content': question})
        answer = pipe_with_kv_cache.generate(question, **generation_config)
        chat_history_with_kv_cache.append({'role': 'assistant', 'content': answer})
        
        chat_history_ov.append({'role': 'user', 'content': question})
        prompt = pipe.get_tokenizer().apply_chat_template(chat_history_ov, add_generation_prompt=True)
        answer = pipe.generate(prompt, **generation_config)
        chat_history_ov.append({'role': 'assistant', 'content': answer})
    pipe_with_kv_cache.finish_chat()

    if chat_history_ov != chat_history_with_kv_cache:
        print(f'kvcache_hist: {chat_history_with_kv_cache}')
        print(f'text_history: {chat_history_ov}')
    assert chat_history_ov == chat_history_with_kv_cache


conversation = [
    {'role': 'user', 'content': '1+1='},
    {'role': 'assistant', 'content': '1 + 1 = 2'},
    {'role': 'user', 'content': 'What is the previous answer?'},
    {'role': 'assistant', 'content': 'The previous answer was: 1 + 1 = 2. Please ask me your next question.'},
    {'role': 'user', 'content': 'Why is the sun yellow?'},
    {'role': 'assistant', 'content': 'Because it emits yeloow light.'},
    {'role': 'user', 'content': 'What was my first question?'},
]
@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize('chat_config', get_chat_templates())
def test_apply_chat_template(model_tmp_path, chat_config: Tuple[str, Dict]):
    tokenizer_config = chat_config[1]

    # Will load openvino_model for tiny-random-phi as a placeholder
    # but indeed only Tokenizer and apply_chat_template will be tested.
    model_id, path, tokenizer, opt_model, pipe = read_model(get_models_list()[0])
    
    full_history_str_hf = tokenizer.apply_chat_template(conversation, 
        add_generation_prompt=False,
        tokenize=False,
        **tokenizer_config)
    
    tok = load_tok([(tokenizer_config, "tokenizer_config.json")], model_tmp_path[1])
    full_history_str = tok.apply_chat_template(conversation, add_generation_prompt=False)
    if full_history_str != full_history_str_hf:
        print(f'hf reference: {full_history_str_hf}')
        print(f'ov_genai out: {full_history_str}')
    assert full_history_str == full_history_str_hf


@pytest.mark.parametrize("generation_config", configs[1:])
@pytest.mark.parametrize("model_descr", get_chat_models_list())
@pytest.mark.precommit
def test_chat_continuous_batching_vs_stateful(model_descr, generation_config: Dict):
    model_id, path, tokenizer, model, stateful = read_model((model_descr[0], model_descr[1] / '_test_chat'))
    cb = get_continuous_batching(path)
    stateful.start_chat()
    cb.start_chat()
    for question in quenstions:
        generated = cb.generate(question, **generation_config)
        reference = stateful.generate(question, **generation_config)
        assert generated == reference
    # Test that finish_chat() doesn't fail just in case.
    cb.finish_chat()

@pytest.mark.precommit
@pytest.mark.nightly
def test_set_chat_template():
    model_descr = get_chat_models_list()[0]
    model_id, path, tokenizer, model_opt, pipe = read_model((model_descr[0], model_descr[1] / '_test_chat'))
    pipe.get_tokenizer().set_chat_template("{% for message in messages %}{{ message['content'] }}{% endfor %}")
    pipe.start_chat()
    generated = pipe.generate("a", max_new_tokens=1)
    pipe.finish_chat()
    reference = pipe.generate("a", max_new_tokens=1)
    assert generated == reference

prompts = [
    '1+1=',
    'What is the previous answer?',
    'Why is the Sun yellow?',
    'What was my first question?',
    ['Why is the Sun yellow?'],
    "若我有一亿美元，在人工智能盛行的今天，我怎样投资才能收益最大化？",
    "מחרוזת בדיקה",
    "Multiline\nstring!\nWow!",
]

@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("add_special_tokens", [True, False])
@pytest.mark.parametrize("prompt", prompts)
def test_add_special_tokens(add_special_tokens, prompt):
    import numpy as np
    model_descr = get_chat_models_list()[0]
    model_id, path, hf_tokenizer, model_opt, pipe = read_model((model_descr[0], model_descr[1] / '_test_chat'))
    genai_tokenzier = pipe.get_tokenizer()
    
    # Calling encode with add_special_tokens will set state flag.
    res_genai = genai_tokenzier.encode(prompt, add_special_tokens).input_ids.data
    res_hf = hf_tokenizer(prompt, return_tensors="np", add_special_tokens=add_special_tokens)["input_ids"]
    assert np.all(res_genai == res_hf)

@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("add_special_tokens", [True, False])
@pytest.mark.parametrize("skip_special_tokens", [True, False])
@pytest.mark.parametrize("prompt", prompts)
def test_add_special_tokens(add_special_tokens, skip_special_tokens, prompt):
    import numpy as np
    model_descr = get_chat_models_list()[0]
    model_id, path, hf_tokenizer, model_opt, pipe = read_model((model_descr[0], model_descr[1] / '_test_chat'))
    genai_tokenizer = pipe.get_tokenizer()
    
    # Calling encode with add_special_tokens will set state flag.
    res_genai = genai_tokenizer.encode(prompt, add_special_tokens).input_ids.data
    res_hf = hf_tokenizer(prompt, return_tensors="np", add_special_tokens=add_special_tokens)["input_ids"]
    assert np.all(res_genai == res_hf)
    
    # Decode with skip_special_tokens
    decoded_genai = genai_tokenizer.decode(res_genai, skip_special_tokens=skip_special_tokens)[0]
    decoded_hf = hf_tokenizer.decode(res_hf[0], skip_special_tokens=skip_special_tokens)
    assert decoded_genai == decoded_hf
