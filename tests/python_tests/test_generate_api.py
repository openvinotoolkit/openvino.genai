# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_genai
from openvino_genai import StopCriteria
import pytest
from list_test_models import models_list
from typing import Union, List, Dict


@pytest.fixture(scope="module", params=models_list(), 
                ids=lambda param: param[0].split('/', 1)[1] if '/' in param[0] else param[0])
def model_fixture(request):
    model_id, path = request.param
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    yield model_id, path, tokenizer, model
    
    import gc
    del tokenizer
    del model
    gc.collect()


def run_hf_ov_genai_comparison_batched(model_fixture, generation_config: Dict, prompts: Union[str, List[str]]):
    model_id, path, tokenizer, model = model_fixture
    device = 'CPU'

    config = generation_config.copy()  # to avoid side effects
    num_beams = config['num_beams'] if 'num_beams' in config else 1

    if not isinstance(prompts, list):
        prompts = [prompts]

    if 'do_sample' not in config:
        # for some reason this key inside HF sometimes is implicitly set to True
        # and it conflicts with beam search args `diversity_penalty` and/or `num_beam_groups` specified here
        # need to state exlicitly to False if not specified
        config['do_sample'] = False

    generation_config_hf = config.copy()
    if generation_config_hf.get('stop_criteria'):
        generation_config_hf['early_stopping'] = stop_criteria_map()[generation_config_hf.pop('stop_criteria')]

    # Encode the batch of prompts
    tokenizer.padding_side = "left"
    encoded_prompts = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    prompt_ids, attention_mask = encoded_prompts['input_ids'], encoded_prompts['attention_mask']
    
    generation_config_hf['num_return_sequences'] = num_beams
    hf_encoded_outputs = model.generate(prompt_ids, attention_mask=attention_mask, 
                                        **generation_config_hf)

    hf_outputs = []
    for idx, hf_encoded_out in enumerate(hf_encoded_outputs):
        prompt_count = idx // num_beams
        hf_outputs.append(tokenizer.decode(hf_encoded_out[prompt_ids[prompt_count].shape[0]:], skip_special_tokens=True))

    import openvino_genai as ov_genai
    pipe = ov_genai.LLMPipeline(path, device)
    
    config['num_return_sequences'] = num_beams * len(prompts)
    ov_outputs = pipe.generate(prompts, **config)
    
    hf_outputs.sort()
    ov_outputs.sort()
    for i, (hf_output, ov_output) in enumerate(zip(hf_outputs, ov_outputs)):
        if hf_output != ov_output:
            print(f'hf_output: {hf_output}')
            print(f'ov_output: {ov_output}')
        assert hf_output == ov_output


def run_hf_ov_genai_comparison(model_fixture, generation_config: Dict, prompt):
    device = 'CPU'
    model_id, path, tokenizer, model = model_fixture

    config = generation_config.copy()  # to avoid side effects

    if 'do_sample' not in config:
        # for some reason this key inside HF sometimes is implicitly set to True
        # and it conflicts with beam search args `diversity_penalty` and/or `num_beam_groups` specified here
        # need to state exlicitly to False if not specified
        config['do_sample'] = False
    
    generation_config_hf = config.copy()
    if generation_config_hf.get('stop_criteria'):
        generation_config_hf['early_stopping'] = stop_criteria_map()[generation_config_hf.pop('stop_criteria')]

    encoded_prompt = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True)
    hf_encoded_output = model.generate(encoded_prompt, **generation_config_hf)
    hf_output = tokenizer.decode(hf_encoded_output[0, encoded_prompt.shape[1]:])

    import openvino_genai as ov_genai
    pipe = ov_genai.LLMPipeline(path, device)
    
    ov_output = pipe.generate(prompt, **config)
    if config.get('num_return_sequences', 1) > 1:
        ov_output = ov_output[0]

    if hf_output != ov_output:
        print(f'hf_output: {hf_output}')
        print(f'ov_output: {ov_output}')

    assert hf_output == ov_output


def stop_criteria_map():
    # in OpenVINO GenAI this parameter is called stop_criteria,
    # while in HF it's called early_stopping. 
    # HF values True, False and "never" correspond to OV GenAI values "EARLY", "HEURISTIC" and "NEVER"
    return {
        StopCriteria.NEVER: "never", 
        StopCriteria.EARLY: True, 
        StopCriteria.HEURISTIC: False
    }


test_cases = [
    (dict(max_new_tokens=20), 'table is made of'),  # generation_config, prompt
    (dict(max_new_tokens=20), '你好！ 你好嗎？'),  # generation_config, prompt
    (dict(num_beam_groups=3, num_beams=15, num_return_sequences=15, max_new_tokens=20, diversity_penalty=1.0), 'Alan Turing was a'),
    (dict(num_beam_groups=3, num_beams=15, num_return_sequences=15, max_new_tokens=30, diversity_penalty=1.0), 'Alan Turing was a'),
    (dict(num_beam_groups=2, num_beams=8, num_return_sequences=8, max_new_tokens=20, diversity_penalty=1.0), 'table is made of'),
    (dict(num_beam_groups=2, num_beams=8, num_return_sequences=8, max_new_tokens=20, diversity_penalty=1.0), 'The Sun is yellow because'),
    (dict(num_beam_groups=2, num_beams=8, num_return_sequences=8, max_new_tokens=20, diversity_penalty=1.5), 'The Sun is yellow because'),
]
@pytest.mark.parametrize("generation_config,prompt", test_cases)
@pytest.mark.precommit
def test_decoding(model_fixture, generation_config, prompt):
    run_hf_ov_genai_comparison(model_fixture, generation_config, prompt)


test_configs = [
    dict(max_new_tokens=20),
    dict( max_new_tokens=20, num_beam_groups=3, num_beams=15,diversity_penalty=1.0)
]
batched_prompts = [['table is made of', 'They sky is blue because', 'Difference between Jupiter and Mars is that'],
                   ['hello', 'Here is the longest nowel ever: '],
                   ['Alan Turing was a', 'return 0', '你好！ 你好嗎？']]
@pytest.mark.parametrize("generation_config", test_configs)
@pytest.mark.parametrize("prompts", batched_prompts)
@pytest.mark.precommit
def test_multibatch(model_fixture, generation_config, prompts):
    generation_config['pad_token_id'] = 2
    run_hf_ov_genai_comparison_batched(model_fixture, generation_config, prompts)


prompts = ['The Sun is yellow because', 'Difference between Jupiter and Mars is that', 'table is made of']
@pytest.mark.parametrize("num_beam_groups", [2, 3, 8])
@pytest.mark.parametrize("group_size", [5, 3, 10])
@pytest.mark.parametrize("max_new_tokens", [20, 15])
@pytest.mark.parametrize("diversity_penalty", [1.0 , 1.5])
@pytest.mark.parametrize("prompt", prompts)
@pytest.mark.precommit
def test_beam_search_decoding(model_fixture, num_beam_groups, group_size, 
                              max_new_tokens, diversity_penalty, prompt):
    generation_config = dict(
        num_beam_groups=num_beam_groups, 
        num_beams=num_beam_groups * group_size, 
        diversity_penalty=diversity_penalty, 
        num_return_sequences=num_beam_groups * group_size, 
        max_new_tokens=max_new_tokens, 
    )
    run_hf_ov_genai_comparison(model_fixture, generation_config, prompt)


@pytest.mark.parametrize("stop_criteria", [StopCriteria.NEVER, StopCriteria.EARLY, StopCriteria.HEURISTIC])
@pytest.mark.parametrize("prompt", prompts)
@pytest.mark.parametrize("max_new_tokens", [10, 80])
@pytest.mark.precommit
def test_stop_criteria(model_fixture, stop_criteria, prompt, max_new_tokens):
    # todo: with EARLY stop_criteria looks like HF return unvalid out with sentence<eos><unk><unk>
    # while genai ends sentence with <eos>
    if (stop_criteria == StopCriteria.EARLY):
        pytest.skip()
    generation_config = dict(
        num_beam_groups=2, 
        num_beams=2 * 3, 
        diversity_penalty=1.0, 
        num_return_sequences=2 * 3, 
        max_new_tokens=max_new_tokens, 
        stop_criteria=stop_criteria,
    )
    run_hf_ov_genai_comparison(model_fixture, generation_config, prompt)


# test long sequences
@pytest.mark.parametrize("num_beam_groups", [2])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("max_new_tokens", [800, 2000])
@pytest.mark.parametrize("prompt", prompts)
@pytest.mark.nightly
def test_beam_search_long_sentences(model_fixture, num_beam_groups, group_size, 
                                    max_new_tokens, prompt):
    generation_config = dict(
        num_beam_groups=num_beam_groups, 
        num_beams=num_beam_groups * group_size, 
        diversity_penalty=1.0, 
        num_return_sequences=num_beam_groups * group_size, 
        max_new_tokens=max_new_tokens, 
    )
    run_hf_ov_genai_comparison(model_fixture, generation_config, prompt)


def user_defined_callback(subword):
    print(subword)


@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
@pytest.mark.precommit
def test_callback_one_string(model_fixture, callback):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    pipe.generate('', openvino_genai.GenerationConfig(), callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
@pytest.mark.precommit
def test_callback_batch_fail(model_fixture, callback):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    with pytest.raises(RuntimeError):
        pipe.generate(['1', '2'], openvino_genai.GenerationConfig(), callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
@pytest.mark.precommit
def test_callback_kwargs_one_string(model_fixture, callback):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    pipe.generate('', max_new_tokens=10, streamer=callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
@pytest.mark.precommit
def test_callback_kwargs_batch_fail(model_fixture, callback):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    with pytest.raises(RuntimeError):
        pipe.generate(['1', '2'], max_new_tokens=10, streamer=callback)


class Printer(openvino_genai.StreamerBase):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    def put(self, token_id):
        # print(self.tokenizer.decode([token_id]))  # Incorrect way to print, but easy to implement
        print(token_id)  # print only token because self.tokenizer.decode([token_id]) are not implemented yet
    def end(self):
        print('end')


@pytest.mark.precommit
def test_streamer_one_string(model_fixture):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    printer = Printer(pipe.get_tokenizer())
    pipe.generate('', openvino_genai.GenerationConfig(), printer)


@pytest.mark.precommit
def test_streamer_batch_fail(model_fixture):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    printer = Printer(pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        pipe.generate(['1', '2'], openvino_genai.GenerationConfig(), printer)


@pytest.mark.precommit
def test_streamer_kwargs_one_string(model_fixture):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    printer = Printer(pipe.get_tokenizer())
    pipe.generate('', do_sample=True, streamer=printer)


@pytest.mark.precommit
def test_streamer_kwargs_batch_fail(model_fixture):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    printer = Printer(pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        pipe.generate('', num_beams=2, streamer=printer)


@pytest.mark.precommit
@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
def test_operator_with_callback_one_string(model_fixture, callback):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    pipe('', openvino_genai.GenerationConfig(), callback)


@pytest.mark.precommit
@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
def test_operator_with_callback_batch_fail(model_fixture, callback):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    with pytest.raises(Exception):
        pipe(['1', '2'], openvino_genai.GenerationConfig(), callback)


@pytest.mark.precommit
def test_operator_with_streamer_kwargs_one_string(model_fixture):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    printer = Printer(pipe.get_tokenizer())
    pipe('', do_sample=True, streamer=printer)


@pytest.mark.precommit
def test_operator_with_streamer_kwargs_batch_fail(model_fixture):
    pipe = openvino_genai.LLMPipeline(model_fixture[1], 'CPU')
    printer = Printer(pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        pipe('', num_beams=2, streamer=printer)
