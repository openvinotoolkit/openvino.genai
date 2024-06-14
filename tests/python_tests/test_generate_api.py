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
from typing import Union, List, Dict, Tuple
import sys
from pathlib import Path
import shutil
import json

@functools.lru_cache(2)
def read_model(params):
    model_id, path = params
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if not (path / 'openvino_model.xml').is_file():
        ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(tokenizer, with_detokenizer=True)
        openvino.save_model(ov_tokenizer, path / "openvino_tokenizer.xml")
        openvino.save_model(ov_detokenizer, path / "openvino_detokenizer.xml")
        
        # to store tokenizer config jsons with special tokens
        tokenizer.save_pretrained(path)
        
        model = optimum.intel.openvino.OVModelForCausalLM.from_pretrained(
            model_id, export=True, trust_remote_code=True,
            compile=False, device='CPU', load_in_8bit=False
        )
        model.generation_config.save_pretrained(path)
        model.config.save_pretrained(path)
        model.save_pretrained(path)
    # Return AutoModelForCausalLM instead of OVModelForCausalLM because
    # there's no way to disable mmap for now. That prohibits the same
    # model from being opened twice at the same time.
    return (
        model_id,
        path,
        tokenizer,
        transformers.AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True),
        openvino_genai.LLMPipeline(str(path)),
    )


def run_hf_ov_genai_comparison_batched(model_descr, generation_config: Dict, prompts: Union[str, List[str]]):
    model_id, path, tokenizer, model, pipe = model_descr
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
    
    hf_encoded_outputs = model.generate(prompt_ids, attention_mask=attention_mask, 
                                        **generation_config_hf)

    hf_outputs = []
    for idx, hf_encoded_out in enumerate(hf_encoded_outputs):
        prompt_count = idx // num_beams
        hf_outputs.append(tokenizer.decode(hf_encoded_out[prompt_ids[prompt_count].shape[0]:], skip_special_tokens=True))

    pipe = openvino_genai.LLMPipeline(str(path))

    ov_outputs = pipe.generate(prompts, **config)
    
    hf_outputs.sort()
    ov_outputs.texts.sort()
    for i, (hf_output, ov_output) in enumerate(zip(hf_outputs, ov_outputs.texts)):
        if hf_output != ov_output:
            print(f'hf_output: {hf_output}')
            print(f'ov_output: {ov_output}')
        assert hf_output == ov_output

def run_hf_ov_genai_comparison(model_descr, generation_config: Dict, prompt):
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
    generation_config_hf.pop('ignore_eos', None)

    encoded_prompt = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True)
    hf_encoded_output = model.generate(encoded_prompt, **generation_config_hf)
    hf_output = tokenizer.decode(hf_encoded_output[0, encoded_prompt.shape[1]:])

    pipe = openvino_genai.LLMPipeline(str(path))

    ov_output = pipe.generate(prompt, **config)
    if config.get('num_return_sequences', 1) > 1:
        assert hf_output in ov_output.texts
    else:
        if hf_output != ov_output.texts:
            print(f'hf_output: {hf_output}')
            print(f'ov_output: {ov_output}')

        assert hf_output == ov_output.texts[0]


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
batched_prompts = [
    ['table is made of', 'They sky is blue because', 'Difference between Jupiter and Mars is that'],
    ['hello', 'Here is the longest nowel ever: '],
    ['Alan Turing was a', 'return 0', '你好！ 你好嗎？']
]
@pytest.mark.parametrize("generation_config", test_configs)
@pytest.mark.parametrize("prompts", batched_prompts)
@pytest.mark.parametrize("model_descr", models_list())
@pytest.mark.precommit
@pytest.mark.xfail(
    raises=AssertionError, reason="assert hf_output == ov_output.texts fails",
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
        # super() may work, but once you begin mixing Python and C++
        # multiple inheritance, things will fall apart due to
        # differences between Python’s MRO and C++’s mechanisms.
        openvino_genai.StreamerBase.__init__(self)
        self.tokenizer = tokenizer
    def put(self, token_id):
        # print(self.tokenizer.decode([token_id]))  # Incorrect way to print, but easy to implement
        print(token_id)  # print only token because self.tokenizer.decode([token_id]) are not implemented yet
    def end(self):
        print('end')


@pytest.mark.precommit
@pytest.mark.xfail(
    raises=RuntimeError, 
    reason="resulting token is out of vocabulary range on Mac",
    strict=False,
    condition=sys.platform == "darwin"
)
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
    pipe.generate('', max_new_tokens=10, do_sample=False, streamer=printer)


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
    return openvino_genai.Tokenizer(str(temp_path))


# load LLMPipline where all configs are cleared
def load_pipe(configs: List[Tuple], temp_path):
    # remove existing jsons from previous tests
    for json_file in temp_path.glob("*.json"):
        json_file.unlink()

    for config_json, config_name in configs:
        with (temp_path / config_name).open('w') as f:
            json.dump(config_json, f)
    return openvino_genai.LLMPipeline(str(temp_path))

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

    config = openvino_genai.GenerationConfig()
    config.do_sample = True  # no eos_token_id but it's loaded from config.json
    pipe.set_generation_config(config)

@pytest.mark.precommit
@pytest.mark.skipif(sys.platform.startswith("win"), reason="probably not enough space for this model on Win")
def test_unicode_pybind_decoding():
    # On this model this prompt generates unfinished utf string.
    # Test that pybind will not fail.
    model_id, path = ("microsoft/phi-1_5", Path("phi-1_5/"))
    pipe = read_model((model_id, path))[4]
    pipe.generate('你好！ 你好嗎？', max_new_tokens=20)


@pytest.skip(reason="probably both models ov + hf doesn't fit to memory")
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

