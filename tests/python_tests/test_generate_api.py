# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools
import openvino
import openvino_tokenizers
import optimum.intel
from openvino_genai import StopCriteria
import openvino_genai as ov_genai
import pytest
import transformers
from list_test_models import models_list, chat_models_list
from typing import Union, List, Dict, Tuple, Optional
import numpy as np
import openvino as ov
import sys
from pathlib import Path
import shutil
import json
import torch


@functools.lru_cache(1)
def read_model(params):
    model_id, path = params
    
    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if path.exists():
        opt_model = OVModelForCausalLM.from_pretrained(path, trust_remote_code=True, 
                                                       compile=False, device='CPU')
    else:
        ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(tokenizer, 
                                                                             add_special_tokens=True, 
                                                                             with_detokenizer=True)
        openvino.save_model(ov_tokenizer, path / "openvino_tokenizer.xml")
        openvino.save_model(ov_detokenizer, path / "openvino_detokenizer.xml")
        
        # to store tokenizer config jsons with special tokens
        tokenizer.save_pretrained(path)
        
        opt_model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True, 
                                                       compile=False, device='CPU', load_in_8bit=False)
        opt_model.generation_config.save_pretrained(path)
        opt_model.config.save_pretrained(path)
        opt_model.save_pretrained(path)
    
    return (
        model_id,
        path,
        tokenizer,
        opt_model,
        ov_genai.LLMPipeline(str(path), device='CPU', config={"ENABLE_MMAP": False}),
    )


def run_hf_ov_genai_comparison_batched(model_descr, generation_config: Dict, prompts: Union[str, List[str]]):
    device = 'CPU'
    model_id, path, tokenizer, model, pipe = model_descr
    config = generation_config.copy()  # to avoid side effects
    num_beams = config['num_beams'] if 'num_beams' in config else 1
    config['num_return_sequences'] = num_beams
    
    if not isinstance(prompts, list):
        prompts = [prompts]

    if 'do_sample' not in config:
        # Some HF models have default do_sample = True, and if we set beam search generation config 
        # it conflicts with `diversity_penalty` and/or `num_beam_groups`.
        # Need to set exlicitly to False, but only if test arguments omitted this arg.
        # Do not apply 'repetition_penalty' if sampling is not used.
        config['do_sample'] = False
        config['repetition_penalty'] = None
    
    generation_config_hf = config.copy()
    if generation_config_hf.get('stop_criteria'):
        generation_config_hf['early_stopping'] = stop_criteria_map()[generation_config_hf.pop('stop_criteria')]
    generation_config_hf.pop('ignore_eos', None)

    # Encode the batch of prompts
    tokenizer.padding_side = "left"
    encoded_prompts = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)
    prompt_ids, attention_mask = encoded_prompts['input_ids'], encoded_prompts['attention_mask']
    
    hf_encoded_outputs = model.generate(prompt_ids, attention_mask=attention_mask, **generation_config_hf)

    hf_outputs = []
    for idx, hf_encoded_out in enumerate(hf_encoded_outputs):
        prompt_count = idx // num_beams
        hf_outputs.append(tokenizer.decode(hf_encoded_out[prompt_ids[prompt_count].shape[0]:], skip_special_tokens=True))

    ov_outputs = pipe.generate(prompts, **config).texts

    hf_outputs.sort()
    ov_outputs.sort()
    for i, (hf_output, ov_output) in enumerate(zip(hf_outputs, ov_outputs)):
        if hf_output != ov_output:
            print(f'hf_output: {hf_output}')
            print(f'ov_output: {ov_output}')
        assert hf_output == ov_output

def run_hf_ov_genai_comparison(model_descr, generation_config: Dict, prompt: str):
    device = 'CPU'
    model_id, path, tokenizer, model, pipe = model_descr

    config = generation_config.copy()  # to avoid side effects

    if 'do_sample' not in config:
        # Some HF models have default do_sample = True, and if we set beam search generation config 
        # it conflicts with `diversity_penalty` and/or `num_beam_groups`.
        # Need to set exlicitly to False, but only if test arguments omitted this arg.
        # Do not apply 'repetition_penalty' if sampling is not used.
        config['do_sample'] = False
        config['repetition_penalty'] = None

    generation_config_hf = config.copy()
    if generation_config_hf.get('stop_criteria'):
        generation_config_hf['early_stopping'] = stop_criteria_map()[generation_config_hf.pop('stop_criteria')]
    generation_config_hf.pop('ignore_eos', None)

    encoded_prompt = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True)
    hf_encoded_output = model.generate(encoded_prompt, **generation_config_hf)
    hf_output = tokenizer.decode(hf_encoded_output[0, encoded_prompt.shape[1]:], skip_special_tokens=True)

    ov_output = pipe.generate(prompt, **config)
    if config.get('num_return_sequences', 1) > 1:
        assert hf_output in ov_output.texts
    else:
        if hf_output != ov_output:
            print(f'hf_output: {hf_output}')
            print(f'ov_output: {ov_output}')

        assert hf_output == ov_output

def hf_ov_genai_tensors_comparison(
        model_descr, 
        generation_config: Dict, 
        input_ids: np.ndarray, 
        attention_mask: Optional[np.array] = None
    ):
    device = 'CPU'
    model_id, path, tokenizer, model, pipe = model_descr

    config = generation_config.copy()  # to avoid side effects

    if 'do_sample' not in config:
        # Some HF models have default do_sample = True, and if we set beam search generation config 
        # it conflicts with `diversity_penalty` and/or `num_beam_groups`.
        # Need to set exlicitly to False, but only if test arguments omitted this arg.
        # Do not apply 'repetition_penalty' if sampling is not used.
        config['do_sample'] = False
        config['repetition_penalty'] = None
    
    generation_config_hf = config.copy()
    if generation_config_hf.get('stop_criteria'):
        generation_config_hf['early_stopping'] = stop_criteria_map()[generation_config_hf.pop('stop_criteria')]
    generation_config_hf.pop('ignore_eos', None)
    
    if attention_mask is not None:
        inputs_ov = ov_genai.TokenizedInputs(ov.Tensor(input_ids), ov.Tensor(attention_mask))
        inputs_hf = dict(inputs=torch.tensor(input_ids), attention_mask=torch.tensor(attention_mask))
    else:
        inputs_hf = dict(inputs=torch.tensor(input_ids))
        inputs_ov = ov.Tensor(input_ids)

    hf_output = model.generate(**inputs_hf, **generation_config_hf)

    pipe = ov_genai.LLMPipeline(str(path), device)
    ov_output = pipe.generate(inputs_ov, **config)

    hf_res = hf_output[0, input_ids.shape[1]:].numpy()
    ov_res = np.array(ov_output.tokens, dtype=np.int64)
    assert np.all(ov_res == hf_res)


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

input_tensors_list = [
    # input_ids, attention_mask
    (np.array([[1, 4, 42]], dtype=np.int64), None),
    (np.array([[1, 4, 42]], dtype=np.int64), np.array([[1, 1, 1]], dtype=np.int64)),
]
@pytest.mark.parametrize("inputs", input_tensors_list)
@pytest.mark.parametrize("model_descr", models_list())
@pytest.mark.xfail(
    raises=TypeError, 
    reason="pybind was unable to find overloads with tensor inputs on Linux",
    strict=False,
    condition=sys.platform == "linux"
)
@pytest.mark.precommit
def test_ov_tensors(model_descr, inputs):
    hf_ov_genai_tensors_comparison(read_model(model_descr), dict(max_new_tokens=20), *inputs)


prompts = [
    'table is made of',
    '你好！ 你好嗎？',
    'Alan Turing was a',
    'The Sun is yellow because',
    ['The Sun is yellow because', 'Alan Turing was a', 'Alan Turing was a']
]
@pytest.mark.parametrize("model_descr", models_list())
@pytest.mark.parametrize("prompt", prompts)
@pytest.mark.precommit
@pytest.mark.xfail(
    raises=TypeError, 
    reason="pybind was unable to find ov::Tensor from openvino yet",
    strict=False,
    condition=sys.platform in ["linux", "win32"]
)
def test_genai_tokenizer_encode(model_descr, prompt):
    model_id, path, tokenizer, model, pipe = read_model(model_descr)
    tok = pipe.get_tokenizer()
    
    encoded_ov = tok.encode(prompt).input_ids.data
    if isinstance(prompt, list):
        encoded_hf = tokenizer.batch_encode_plus(prompt)['input_ids']
        for tokens_ov, tokens_hf in zip(encoded_ov, encoded_hf):
            assert np.all(tokens_ov == tokens_hf)
    else:
        encoded_hf = tokenizer.encode(prompt)
        assert np.all(encoded_hf == encoded_ov[0])

encoded_prompts = [
    [1, 1591, 338, 1754, 310],
    [1, 17102,   323,  3864,   471,   263],
    
    # chineze characters
    [1, 29871, 30919, 31076, 30584, 29871, 30919, 31076, 232, 154, 145, 30882],

    # On meta-llama/Meta-Llama-3-8B-Instruct this becomes longer  after removing the last token
    [3113, 264, 364, 267],

    # batched tokens
    [[1, 1591, 338, 1754, 310], [1, 1591, 338, 1754, 310], [1, 17102,   323,  3864,   471,   263]]
]
@pytest.mark.parametrize("model_descr", models_list())
@pytest.mark.parametrize("encoded_prompt", encoded_prompts)
@pytest.mark.precommit
@pytest.mark.xfail(
    raises=TypeError, 
    reason="pybind was unable to find ov::Tensor from openvino yet",
    strict=False,
    condition=sys.platform in ["linux", "win32"]
)
def test_genai_tokenizer_decode(model_descr, encoded_prompt):
    model_id, path, tokenizer, model, pipe = read_model(model_descr)
    tok = pipe.get_tokenizer()
    decoded_ov = tok.decode(encoded_prompt)
    
    if isinstance(encoded_prompt[0], list):
        decoded_hf = tokenizer.batch_decode(encoded_prompt, skip_special_tokens=True)
        for tokens_ov, tokens_hf in zip(decoded_ov, decoded_hf):
            assert np.all(tokens_ov == tokens_hf)
    else:
        decoded_hf = tokenizer.decode(encoded_prompt, skip_special_tokens=True)
        assert decoded_hf == decoded_ov


test_configs = [
    dict(max_new_tokens=20),
    dict(max_new_tokens=200, ignore_eos=True),
    dict(max_new_tokens=20, num_beam_groups=3, num_beams=15, diversity_penalty=1.0)
]
batched_prompts = [
    ['table is made', 'They sky is blue because', 'Difference between Jupiter and Mars is that'],
    ['hello', 'Here is the longest nowel ever: '],
    ['Alan Turing was a', 'return 0', '你好！ 你好嗎？'],
    ['table is made', 'table is made [force left pad tokens]']
]
@pytest.mark.parametrize("generation_config", test_configs)
@pytest.mark.parametrize("prompts", batched_prompts)
@pytest.mark.parametrize("model_descr", models_list())
@pytest.mark.precommit
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
    pipe.generate('table is made of', generation_config, callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
@pytest.mark.precommit
def test_callback_batch_fail(callback):
    pipe = read_model(models_list()[0])[4]
    with pytest.raises(RuntimeError):
        pipe.generate(['1', '2'], ov_genai.GenerationConfig(), callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
@pytest.mark.precommit
def test_callback_kwargs_one_string(callback):
    pipe = read_model(models_list()[0])[4]
    pipe.generate('table is made of', max_new_tokens=10, streamer=callback)

@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
@pytest.mark.precommit
@pytest.mark.parametrize("model_descr", models_list())
def test_callback_decoding_metallama(model_descr, callback):
    # On metallam this prompt generates output which can shorten after adding new tokens.
    # Test that streamer correctly handles such cases.
    prompt = 'I have an interview about product speccing with the company Weekend Health. Give me an example of a question they might ask with regards about a new feature'
    if model_descr[0] != 'meta-llama/Meta-Llama-3-8B-Instruct':
        pytest.skip()
    pipe = read_model(model_descr)[4]
    pipe.generate(prompt, max_new_tokens=300, streamer=callback)


@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
@pytest.mark.precommit
def test_callback_kwargs_batch_fail(callback):
    pipe = read_model(models_list()[0])[4]
    with pytest.raises(RuntimeError):
        pipe.generate(['1', '2'], max_new_tokens=10, streamer=callback)


class Printer(ov_genai.StreamerBase):
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


@pytest.mark.precommit
def test_streamer_one_string():
    pipe = read_model(models_list()[0])[4]
    generation_config = pipe.get_generation_config()
    generation_config.max_new_tokens = 10
    printer = Printer(pipe.get_tokenizer())
    pipe.generate('table is made of', generation_config, printer)


@pytest.mark.precommit
def test_streamer_batch_fail():
    pipe = read_model(models_list()[0])[4]
    printer = Printer(pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        pipe.generate(['1', '2'], ov_genai.GenerationConfig(), printer)


@pytest.mark.precommit
def test_streamer_kwargs_one_string():
    pipe = read_model(models_list()[0])[4]
    printer = Printer(pipe.get_tokenizer())
    pipe.generate('table is made of', max_new_tokens=10, do_sample=False, streamer=printer)


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
    pipe('talbe is made of', ten_tokens, callback)


@pytest.mark.precommit
@pytest.mark.parametrize("callback", [print, user_defined_callback, lambda subword: print(subword)])
def test_operator_with_callback_batch_fail(callback):
    pipe = read_model(models_list()[0])[4]
    with pytest.raises(RuntimeError):
        pipe(['1', '2'], ov_genai.GenerationConfig(), callback)


@pytest.mark.precommit
def test_operator_with_streamer_kwargs_one_string():
    pipe = read_model(models_list()[0])[4]
    printer = Printer(pipe.get_tokenizer())
    pipe('hi', max_new_tokens=10, do_sample=True, streamer=printer)


@pytest.mark.precommit
def test_operator_with_streamer_kwargs_batch_fail():
    pipe = read_model(models_list()[0])[4]
    printer = Printer(pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        pipe('', num_beams=2, streamer=printer)


@pytest.fixture(scope="module")
def model_tmp_path(tmpdir_factory):
    model_id, path, _, _, _ = read_model(models_list()[0])
    temp_path = tmpdir_factory.mktemp(model_id.replace('/', '_'))

    # copy openvino converted model and tokenizers
    for pattern in ['*.xml', '*.bin']:
        for src_file in path.glob(pattern):
            if src_file.is_file():
                shutil.copy(src_file, temp_path / src_file.name)    
    yield model_id, Path(temp_path)


# load Tokenizer where all configs are cleared
def load_tok(configs: List[Tuple], temp_path):
    # remove existing jsons from previous tests
    for json_file in temp_path.glob("*.json"):
        json_file.unlink()

    for config_json, config_name in configs:
        with (temp_path / config_name).open('w') as f:
            json.dump(config_json, f)
    return ov_genai.Tokenizer(str(temp_path))


# load LLMPipline where all configs are cleared
def load_pipe(configs: List[Tuple], temp_path):
    # remove existing jsons from previous tests
    for json_file in temp_path.glob("*.json"):
        json_file.unlink()

    for config_json, config_name in configs:
        with (temp_path / config_name).open('w') as f:
            json.dump(config_json, f)
    return ov_genai.LLMPipeline(str(temp_path))

@pytest.mark.precommit
def test_load_special_tokens_ids_1(model_tmp_path):
    # test when there is an available config.json
    config_json = { 
        "pad_token_id": 422,
        "bos_token_id": 42, 
        "eos_token_id": 37,
    }
    tok = load_tok([(config_json, "config.json")], model_tmp_path[1])
    assert tok.get_pad_token_id() == config_json['pad_token_id']
    assert tok.get_bos_token_id() == config_json['bos_token_id']
    assert tok.get_eos_token_id() == config_json['eos_token_id']


@pytest.mark.precommit
def test_load_special_tokens_str_2(model_tmp_path):
    # test with special_tokens_map
    special_tokens_map_json = { 
        "pad_token": {"content": "<custom_pad>"},
        "bos_token": {"content": "<custom_bos>"},
        "eos_token": {"content": "<custom_eos>"},
    }
    tok = load_tok([(special_tokens_map_json, "special_tokens_map.json")], model_tmp_path[1])
    assert tok.get_pad_token() == special_tokens_map_json['pad_token']["content"]
    assert tok.get_bos_token() == special_tokens_map_json['bos_token']["content"]
    assert tok.get_eos_token() == special_tokens_map_json['eos_token']["content"]


@pytest.mark.precommit
def test_load_special_tokens_3_(model_tmp_path):
    # special_tokens_map is not available 
    # but tokenize_config.json exists
    # will load both string and integer representations
    tok_config_json = {
        "added_tokens_decoder": {
            "422": {"content": "<pad>"},
            "37": {"content": "<s>"},
            "42": {"content": "</s>"},
        },
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "eos_token": "</s>",
    }

    tok = load_tok([(tok_config_json, "tokenizer_config.json")], model_tmp_path[1])
    assert tok.get_pad_token() == tok_config_json['pad_token']
    assert tok.get_bos_token() == tok_config_json['bos_token']
    assert tok.get_eos_token() == tok_config_json['eos_token']

    assert tok.get_pad_token_id() == 422
    assert tok.get_bos_token_id() == 37
    assert tok.get_eos_token_id() == 42


@pytest.mark.precommit
def test_load_special_tokens_3(model_tmp_path):
    # both config.json is availabel and tokenizer_config.json available
    # check that it does not read int values from tokenizer_config.json if they are in config.json
    tok_config_json = {
    "added_tokens_decoder": {
        # integers differ from config.json to check they don't override config.json
        "777": {"content": "<pad>"},
        "888": {"content": "<s>"},
        "656": {"content": "</s>"},
    },
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    }
    config_json = { 
        "pad_token_id": 422,
        "bos_token_id": 42, 
        "eos_token_id": 37,
    }
    configs = [
        (tok_config_json, "tokenizer_config.json"),
        (config_json, "config.json")
    ]
    tok = load_tok(configs, model_tmp_path[1])
    assert tok.get_pad_token_id() == config_json['pad_token_id']
    assert tok.get_bos_token_id() == config_json['bos_token_id']
    assert tok.get_eos_token_id() == config_json['eos_token_id']

    assert tok.get_pad_token() == tok_config_json['pad_token']
    assert tok.get_bos_token() == tok_config_json['bos_token']
    assert tok.get_eos_token() == tok_config_json['eos_token']


@pytest.mark.precommit
@pytest.mark.xfail(
    raises=AssertionError, 
    reason="CVS-143410 ov tokenizer should be aligned with hf",
    strict=False,
)
def test_load_special_tokens_4(model_tmp_path):
    # only string representation is provided, find token integers by inference
    model_id, temp_path = model_tmp_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    special_tokens_map_json = {}
    token_str_int_map = {}
    special_token_names = ['pad_token', 'bos_token', 'eos_token']
    for token_str in special_token_names:
        if hasattr(tokenizer, token_str):
            token_val = getattr(tokenizer, token_str)
            special_tokens_map_json.update({token_str: {"content": token_val}})
            token_id = tokenizer(token_val, add_special_tokens=False)['input_ids'][0]
            token_str_int_map.update({token_str: token_id})

    # since only string representations are present in the json will try to get by inference
    tok = load_tok([(special_tokens_map_json, "special_tokens_map.json")], temp_path)

    # check ids inferred correctly for special tokens existing if HF tokenizer
    if 'pad_token' in token_str_int_map:
        assert tok.get_pad_token_id() == token_str_int_map['pad_token']
    if 'bos_token' in token_str_int_map:
        assert tok.get_bos_token_id() == token_str_int_map['bos_token']
    if 'eos_token' in token_str_int_map:
        assert tok.get_eos_token_id() == token_str_int_map['eos_token']


invalid_configs = [
    dict(num_beam_groups=3, num_beams=15, do_sample=True),
    dict(do_sample=True),  # no eos_token_id no max_new_tokens, no max_len
    dict(eos_token_id=42, ignore_eos=True),  # no max_new_tokens, no max_len with ignore_eos
    dict(repetition_penalty=-1.0, eos_token_id=42, max_new_tokens=20), # invalid penalty
    dict(temperature=-1.0, do_sample=True, eos_token_id=42, max_new_tokens=20), # invalid temp
    dict(top_p=-1.0, do_sample=True, eos_token_id=42, max_new_tokens=20), # invalid top_p
    dict(top_k=0, do_sample=True, eos_token_id=42, max_new_tokens=20), # invalid top_k
]
@pytest.mark.parametrize("generation_config", invalid_configs)
@pytest.mark.precommit
def test_invalid_configs(model_tmp_path, generation_config):
    model_id, temp_path = model_tmp_path
    config_json = {}
    pipe = load_pipe([(config_json, "config.json")], temp_path)
    with pytest.raises(RuntimeError):
        pipe.generate('blah blah', **generation_config)


@pytest.mark.precommit
def test_valid_configs(model_tmp_path):
    model_id, temp_path = model_tmp_path
    pipe = load_pipe([({"eos_token_id": 37}, "config.json")], temp_path)

    config = ov_genai.GenerationConfig()
    config.do_sample = True  # no eos_token_id but it's loaded from config.json
    pipe.set_generation_config(config)

invalid_py_configs = [
    dict(num_beam_groups=3, num_beams=15, do_sample=True),
    dict(unexisting_key_name=True),  # no eos_token_id no max_new_tokens, no max_len
    dict(eos_token_id=42, ignore_eos=True),  # no max_new_tokens, no max_len with ignore_eos
    dict(repetition_penalty=-1.0, eos_token_id=42, max_new_tokens=20), # invalid penalty
    dict(temperature=-1.0, do_sample=True, eos_token_id=42, max_new_tokens=20), # invalid temp
    dict(top_p=-1.0, do_sample=True, eos_token_id=42, max_new_tokens=20), # invalid top_p
    dict(top_k=0, do_sample=True, eos_token_id=42, max_new_tokens=20), # invalid top_k
]
@pytest.mark.precommit
@pytest.mark.parametrize("generation_config", invalid_py_configs)
def test_python_generation_config_validation(model_tmp_path, generation_config):
    model_id, temp_path = model_tmp_path
    pipe = load_pipe([({"eos_token_id": 37}, "config.json")], temp_path)
    
    # 'unexisting_key_name' key validity is checked in pybind and ValueError will be returned
    #  instead of RuntimeError, which is returned when GenerationConfig values are validated
    return_exception_type = ValueError if 'unexisting_key_name' in generation_config else RuntimeError
    with pytest.raises(return_exception_type):
        pipe.set_generation_config(ov_genai.GenerationConfig(**generation_config))


@pytest.mark.precommit
@pytest.mark.skipif(sys.platform.startswith("win"), reason="probably not enough space for this model on Win")
def test_unicode_pybind_decoding_1():
    # On this model this prompt generates unfinished utf string.
    # Test that pybind will not fail.
    model_id, path = ("microsoft/phi-1_5", Path("phi-1_5/"))
    pipe = read_model((model_id, path))[4]
    res_str = pipe.generate('你好！ 你好嗎？', max_new_tokens=20)
    assert isinstance(res_str, str)
    assert len(res_str) > 0


@pytest.mark.precommit
@pytest.mark.skipif(sys.platform.startswith("win"), reason="probably not enough space for this model on Win")
def test_unicode_pybind_decoding_2():
    # On this model this prompt generates unfinished utf string.
    # Test that pybind will not fail.
    model_id, path = ("microsoft/phi-1_5", Path("phi-1_5/"))
    pipe = read_model((model_id, path))[4]
    decoded_results = pipe.generate(['你好！ 你好嗎？'], max_new_tokens=20)
    assert isinstance(decoded_results, ov_genai.DecodedResults)
    assert len(decoded_results.texts[0]) > 0


@pytest.mark.precommit
@pytest.mark.skipif(sys.platform.startswith("win"), reason="probably not enough space for this model on Win")
def test_unicode_pybind_decoding_3():
    # On this model this prompt generates unfinished utf-8 string
    # and streams it. Test that pybind will not fail while we pass string to python.
    model_id, path = ("microsoft/phi-1_5", Path("phi-1_5/"))
    pipe = read_model((model_id, path))[4]
    pipe.generate('你好！ 你好嗎？', max_new_tokens=20, streamer=lambda x: print(x))


quenstions = [
    '1+1=',
    'What is the previous answer?',
    'Why is the sun yellow?',
    'What was my first question?'
]

configs = [
    dict(max_new_tokens=500),
    # dict(num_beam_groups=3, num_beams=15, num_return_sequences=15, max_new_tokens=20, diversity_penalty=1.0)
]
@pytest.mark.parametrize("generation_config", configs)
@pytest.mark.parametrize("model_descr", chat_models_list())
@pytest.mark.precommit
@pytest.mark.skipif(sys.platform == "linux", reason="no space left on linux device for chat models")
def test_chat_1(model_descr, generation_config):
    config = generation_config.copy()  # to avoid side effects
    
    if 'do_sample' not in config:
        # Some HF models have default do_sample = True, and if we set beam search generation config 
        # it conflicts with `diversity_penalty` and/or `num_beam_groups`.
        # Need to set exlicitly to False, but only if test arguments omitted this arg.
        # Do not apply 'repetition_penalty' if sampling is not used.
        config['do_sample'] = False
        config['repetition_penalty'] = None
    
    config_hf = config.copy()
    if config_hf.get('stop_criteria'):
        config_hf['early_stopping'] = stop_criteria_map()[config_hf.pop('stop_criteria')]
    config_hf.pop('ignore_eos', None)

    chat_history_hf = []
    chat_history_ov = []
    chat_prompt = ''
    model_id, path, tokenizer, model_opt, pipe = read_model(model_descr)
    ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(tokenizer, add_special_tokens=False, with_detokenizer=True)
    openvino.save_model(ov_tokenizer, path / "openvino_tokenizer.xml")
    openvino.save_model(ov_detokenizer, path / "openvino_detokenizer.xml")
    ov_genai.LLMPipeline(str(path), device='CPU', config={"ENABLE_MMAP": False})

    pipe.start_chat()    
    for prompt in quenstions:
        chat_history_hf.append({'role': 'user', 'content': prompt})
        chat_history_ov.append({'role': 'user', 'content': prompt})
        
        chat_prompt = tokenizer.apply_chat_template(chat_history_hf, tokenize=False, add_generation_prompt=True)
        tokenized = tokenizer(chat_prompt, return_tensors='pt', add_special_tokens=False)
        
        answer = model_opt.generate(**tokenized, **config_hf)
        answer_str = tokenizer.decode(answer[0, tokenized['input_ids'].numel():], skip_special_tokens=True)
        chat_history_hf.append({'role': 'assistant', 'content': answer_str})

        answer_ov = pipe.generate(prompt, **config)
        chat_history_ov.append({'role': 'assistant', 'content': answer_ov})

    pipe.finish_chat()
    
    if chat_history_ov != chat_history_hf:
        print(f'hf_output: {chat_history_hf}')
        print(f'ov_output: {chat_history_ov}')
    assert chat_history_ov == chat_history_hf
    pipe.generate('你好！ 你好嗎？', max_new_tokens=20)


@pytest.mark.skip(reason="probably both models ov + hf doesn't fit to memory")
@pytest.mark.precommit
@pytest.mark.skipif(sys.platform.startswith("win"), reason="probably not enough space for this model on Win")
def test_left_pad():
    # test left pad tokenizer post processing implementation
    prompts = [
        "The Sun is yellow because",
        "The Sun is yellow because [force left pad tokens]"
    ]
    models = read_model(("microsoft/phi-1_5", Path("phi-1_5/")))

    config = {
        "max_new_tokens": 20,
        "num_beam_groups": 2,
        "num_beams": 2,
        "num_return_sequences": 2,
        "do_sample": False,
        "diversity_penalty": 1.0,
        # phi 1_5 has no eos_token_id in model configuration
        # ov genai will detect eos_token_id from tokenizer config
        # hf implementation doesn't fetch it from tokenizer config and defaults to None
        # align ov genai and hf by setting eos_token_id explicitly
        "eos_token_id": 50256,
    }

    models[2].pad_token = models[2].eos_token
    run_hf_ov_genai_comparison_batched(models, config, prompts)
