# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools
import openvino
import openvino_genai
import openvino_tokenizers
import optimum.intel
from openvino_genai import StopCriteria
import pytest
import transformers
from list_test_models import models_list
from typing import Union, List, Dict


@functools.lru_cache(1)
def read_model(params):
    model_id, path = params
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    if not (path / 'openvino_model.xml').is_file():
        ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(tokenizer, with_detokenizer=True)
        openvino.save_model(ov_tokenizer, path / "openvino_tokenizer.xml")
        openvino.save_model(ov_detokenizer, path / "openvino_detokenizer.xml")
        optimum.intel.openvino.OVModelForCausalLM.from_pretrained(
            model_id, export=True, trust_remote_code=True,
            compile=False, device='CPU', load_in_8bit=False
        ).save_pretrained(path)
    # Return AutoModelForCausalLM instead of OVModelForCausalLM because
    # there's no way to disable mmap for now. That prohibits the same
    # model from being opened twice at the same time.
    return (
        model_id,
        path,
        tokenizer,
        transformers.AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True),
        openvino_genai.LLMPipeline(str(path)),
    )


def run_hf_ov_genai_comparison_batched(model_descr, generation_config: Dict, prompts: Union[str, List[str]]):
    model_id, path, tokenizer, model, pipe = model_descr
    device = 'CPU'

    config = generation_config.copy()  # to avoid side effects
    num_beams = config['num_beams'] if 'num_beams' in config else 1

    if not isinstance(prompts, list):
        prompts = [prompts]

    if 'do_sample' not in config:
        # Some HF model has default do_sample = True, and if we test beam search
        # it conflicts with `diversity_penalty` and/or `num_beam_groups`.
        # Need to set exlicitly to False, but only if test arguments omitted this arg.
        config['do_sample'] = False
    
    generation_config_hf = config.copy()
    if generation_config_hf.get('stop_criteria'):
        generation_config_hf['early_stopping'] = stop_criteria_map()[generation_config_hf.pop('stop_criteria')]
    generation_config_hf.pop('ignore_eos', None)

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
    pipe = ov_genai.LLMPipeline(str(path), device)
    
    config['num_return_sequences'] = num_beams * len(prompts)
    ov_outputs = pipe.generate(prompts, **config)
    
    hf_outputs.sort()
    ov_outputs.sort()
    for i, (hf_output, ov_output) in enumerate(zip(hf_outputs, ov_outputs)):
        if hf_output != ov_output:
            print(f'hf_output: {hf_output}')
            print(f'ov_output: {ov_output}')
        assert hf_output == ov_output

def run_hf_ov_genai_comparison(model_descr, generation_config: Dict, prompt):
    device = 'CPU'
    model_id, path, tokenizer, model, pipe = model_descr

    config = generation_config.copy()  # to avoid side effects

    if 'do_sample' not in config:
        # Some HF model has default do_sample = True, and if we test beam search
        # it conflicts with `diversity_penalty` and/or `num_beam_groups`.
        # Need to set exlicitly to False, but only if test arguments omitted this arg.
        config['do_sample'] = False
    
    generation_config_hf = config.copy()
    if generation_config_hf.get('stop_criteria'):
        generation_config_hf['early_stopping'] = stop_criteria_map()[generation_config_hf.pop('stop_criteria')]

    encoded_prompt = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True)
    hf_encoded_output = model.generate(encoded_prompt, **generation_config_hf)
    hf_output = tokenizer.decode(hf_encoded_output[0, encoded_prompt.shape[1]:])

    import openvino_genai as ov_genai
    pipe = ov_genai.LLMPipeline(str(path), device)
    
    ov_output = pipe.generate(prompt, **config)
    if config.get('num_return_sequences', 1) > 1:
        assert hf_output in ov_output
    else:
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
    (dict(max_new_tokens=20), 'table is made of'),
    (dict(max_new_tokens=20), '你好！ 你好嗎？'),
    (dict(num_beam_groups=3, num_beams=15, num_return_sequences=15, max_new_tokens=20, diversity_penalty=1.0), 'Alan Turing was a'),
    (dict(num_beam_groups=3, num_beams=15, num_return_sequences=15, max_new_tokens=30, diversity_penalty=1.0), 'Alan Turing was a'),
    (dict(num_beam_groups=2, num_beams=8, num_return_sequences=8, max_new_tokens=20, diversity_penalty=1.0), 'table is made of'),
    (dict(num_beam_groups=2, num_beams=8, num_return_sequences=8, max_new_tokens=20, diversity_penalty=1.0), 'The Sun is yellow because'),
    (dict(num_beam_groups=2, num_beams=8, num_return_sequences=8, max_new_tokens=20, diversity_penalty=1.5), 'The Sun is yellow because'),
]
@pytest.mark.parametrize("generation_config,prompt", test_cases)
@pytest.mark.parametrize("model_descr", models_list())
@pytest.mark.precommit
def test_decoding(model_descr, generation_config, prompt):
    run_hf_ov_genai_comparison(read_model(model_descr), generation_config, prompt)


test_configs = [
    dict(max_new_tokens=20),
    dict(max_new_tokens=200, ignore_eos=True),
    dict(max_new_tokens=20, num_beam_groups=3, num_beams=15, diversity_penalty=1.0)
]
batched_prompts = [['table is made of', 'They sky is blue because', 'Difference between Jupiter and Mars is that'],
                   ['hello', 'Here is the longest nowel ever: '],
                   ['Alan Turing was a', 'return 0', '你好！ 你好嗎？']]
@pytest.mark.parametrize("generation_config", test_configs)
@pytest.mark.parametrize("prompts", batched_prompts)
@pytest.mark.parametrize("model_descr", models_list())
@pytest.mark.precommit
@pytest.mark.xfail(
    raises=AssertionError, reason="assert hf_output == ov_output fails",
    strict=False,
)
def test_multibatch(model_descr, generation_config, prompts):
    run_hf_ov_genai_comparison_batched(read_model(model_descr), generation_config, prompts)


prompts = ['The Sun is yellow because', 'Difference between Jupiter and Mars is that', 'table is made of']
@pytest.mark.parametrize("num_beam_groups", [2, 3, 8])
@pytest.mark.parametrize("group_size", [5, 3, 10])
@pytest.mark.parametrize("max_new_tokens", [20, 15])
@pytest.mark.parametrize("diversity_penalty", [1.0 , 1.5])
@pytest.mark.parametrize("prompt", prompts)
@pytest.mark.parametrize("model_descr", models_list())
@pytest.mark.precommit
def test_beam_search_decoding(model_descr, num_beam_groups, group_size,
                              max_new_tokens, diversity_penalty, prompt):
    generation_config = dict(
        num_beam_groups=num_beam_groups, 
        num_beams=num_beam_groups * group_size, 
        diversity_penalty=diversity_penalty, 
        num_return_sequences=num_beam_groups * group_size, 
        max_new_tokens=max_new_tokens, 
    )
    run_hf_ov_genai_comparison(read_model(model_descr), generation_config, prompt)


@pytest.mark.parametrize("stop_criteria", [StopCriteria.NEVER, StopCriteria.EARLY, StopCriteria.HEURISTIC])
@pytest.mark.parametrize("prompt", prompts)
@pytest.mark.parametrize("max_new_tokens", [10, 80])
@pytest.mark.parametrize("model_descr", models_list())
@pytest.mark.precommit
def test_stop_criteria(model_descr, stop_criteria, prompt, max_new_tokens):
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
    run_hf_ov_genai_comparison(read_model(model_descr), generation_config, prompt)


# test long sequences
@pytest.mark.parametrize("num_beam_groups", [2])
@pytest.mark.parametrize("group_size", [5])
@pytest.mark.parametrize("max_new_tokens", [800, 2000])
@pytest.mark.parametrize("prompt", prompts)
@pytest.mark.parametrize("model_descr", models_list())
@pytest.mark.skip(reason="Will be enabled in nightly since the test are computationally expensive")
@pytest.mark.nightly
def test_beam_search_long_sentences(model_descr, num_beam_groups, group_size,
                                    max_new_tokens, prompt):
    generation_config = dict(
        num_beam_groups=num_beam_groups, 
        num_beams=num_beam_groups * group_size, 
        diversity_penalty=1.0, 
        num_return_sequences=num_beam_groups * group_size, 
        max_new_tokens=max_new_tokens, 
    )
    run_hf_ov_genai_comparison(read_model(model_descr), generation_config, prompt)


def user_defined_callback(subword):
    print(subword)


@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
@pytest.mark.precommit
def test_callback_one_string(callback):
    pipe = read_model(models_list()[0])[4]
    generation_config = pipe.get_generation_config()
    generation_config.max_new_tokens = 10
    pipe.generate('', generation_config, callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
@pytest.mark.precommit
def test_callback_batch_fail(callback):
    pipe = read_model(models_list()[0])[4]
    with pytest.raises(RuntimeError):
        pipe.generate(['1', '2'], openvino_genai.GenerationConfig(), callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
@pytest.mark.precommit
def test_callback_kwargs_one_string(callback):
    pipe = read_model(models_list()[0])[4]
    pipe.generate('', max_new_tokens=10, streamer=callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
@pytest.mark.precommit
def test_callback_kwargs_batch_fail(callback):
    pipe = read_model(models_list()[0])[4]
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
def test_streamer_one_string():
    pipe = read_model(models_list()[0])[4]
    generation_config = pipe.get_generation_config()
    generation_config.max_new_tokens = 10
    printer = Printer(pipe.get_tokenizer())
    pipe.generate('', generation_config, printer)


@pytest.mark.precommit
def test_streamer_batch_fail():
    pipe = read_model(models_list()[0])[4]
    printer = Printer(pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        pipe.generate(['1', '2'], openvino_genai.GenerationConfig(), printer)


@pytest.mark.precommit
def test_streamer_kwargs_one_string():
    pipe = read_model(models_list()[0])[4]
    printer = Printer(pipe.get_tokenizer())
    pipe.generate('', max_new_tokens=10, do_sample=True, streamer=printer)


@pytest.mark.precommit
def test_streamer_kwargs_batch_fail():
    pipe = read_model(models_list()[0])[4]
    printer = Printer(pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        pipe.generate('', num_beams=2, streamer=printer)


@pytest.mark.precommit
@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
def test_operator_with_callback_one_string(callback):
    pipe = read_model(models_list()[0])[4]
    ten_tokens = pipe.get_generation_config()
    ten_tokens.max_new_tokens = 10
    pipe('', ten_tokens, callback)


@pytest.mark.precommit
@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
def test_operator_with_callback_batch_fail(callback):
    pipe = read_model(models_list()[0])[4]
    with pytest.raises(TypeError):
        pipe(['1', '2'], openvino_genai.GenerationConfig(), callback)


@pytest.mark.precommit
def test_operator_with_streamer_kwargs_one_string():
    pipe = read_model(models_list()[0])[4]
    printer = Printer(pipe.get_tokenizer())
    pipe('', max_new_tokens=10, do_sample=True, streamer=printer)


@pytest.mark.precommit
def test_operator_with_streamer_kwargs_batch_fail():
    pipe = read_model(models_list()[0])[4]
    printer = Printer(pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        pipe('', num_beams=2, streamer=printer)
