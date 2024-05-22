# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from list_test_models import models_list


@pytest.fixture(scope="module", params=models_list())
def model_fixture(request):
    model_id, path = request.param
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return model_id, path, tokenizer, model

def run_hf_ov_genai_comparison(model_fixture, generation_config, prompt):
    model_id, path, tokenizer, model = model_fixture

    generation_config_hf = generation_config.copy()
    # in OpenVINO GenAI this parameter is called stop_criteria,
    # while in HF it's called early_stopping. 
    # HF values True, False and "never" correspond to OV GenAI values "early", "heuristic" and "never"
    if generation_config_hf.get('stop_criteria'):
        generation_config_hf['early_stopping'] = stop_criteria_map()[generation_config_hf.pop('stop_criteria')]

    encoded_prompt = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True)
    hf_encoded_output = model.generate(encoded_prompt, **generation_config_hf)
    hf_output = tokenizer.decode(hf_encoded_output[0, encoded_prompt.shape[1]:])

    device = 'CPU'
    ov_tokenizers_path = '../../build/openvino_tokenizers/src/'
    import openvino_genai as ov_genai
    
    pipe = ov_genai.LLMPipeline(path, device, {}, ov_tokenizers_path)
    ov_output = pipe.generate(prompt, **generation_config)

    if hf_output != ov_output:
        print(f'hf_output: {hf_output}')
        print(f'ov_output: {ov_output}')

    assert hf_output == ov_output


def stop_criteria_map():
    return {"never": "never", "early": True, "heuristic": False}

test_cases = [
    (dict(max_new_tokens=20, do_sample=False), 'table is made of'),  # generation_config, prompt
    # (dict(num_beam_groups=3, num_beams=15, num_return_sequences=15, max_new_tokens=20, diversity_penalty=1.0), 'table is made of'),
    # (dict(num_beam_groups=3, num_beams=15, num_return_sequences=15, max_new_tokens=20, diversity_penalty=1.0), 'Alan Turing was a'),
    # (dict(num_beam_groups=3, num_beams=15, num_return_sequences=15, max_new_tokens=30, diversity_penalty=1.0), 'Alan Turing was a'),
    # (dict(num_beam_groups=2, num_beams=8, num_return_sequences=8, max_new_tokens=20, diversity_penalty=1.0), 'table is made of'),
    # (dict(num_beam_groups=2, num_beams=8, num_return_sequences=8, max_new_tokens=20, diversity_penalty=1.0), 'The Sun is yellow because'),
    # (dict(num_beam_groups=2, num_beams=8, num_return_sequences=8, max_new_tokens=20, diversity_penalty=1.5), 'The Sun is yellow because'),
]
@pytest.mark.parametrize("generation_config,prompt", test_cases)
def test_greedy_decoding(model_fixture, generation_config, prompt):
    run_hf_ov_genai_comparison(model_fixture, generation_config, prompt)


prompts = ['The Sun is yellow because', 'Alan Turing was a', 'table is made of']
@pytest.mark.parametrize("num_beam_groups", [2, 3, 8])
@pytest.mark.parametrize("group_size", [5, 3, 10])
@pytest.mark.parametrize("max_new_tokens", [20, 15])
@pytest.mark.parametrize("diversity_penalty", [1.0, 1.5])
@pytest.mark.parametrize("prompt", prompts)
@pytest.mark.skip  # temporarily
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


@pytest.mark.parametrize("stop_criteria", ["never", "early", "heuristic"])
@pytest.mark.parametrize("prompt", prompts)
@pytest.mark.parametrize("max_new_tokens", [20, 40, 300])
@pytest.mark.skip # temporarily
def test_stop_criteria(model_fixture, stop_criteria, prompt, max_new_tokens):
    # todo: for long sentences early stop_criteria fails
    if (stop_criteria == 'early' and max_new_tokens >= 300):
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
@pytest.mark.parametrize("diversity_penalty", [1.0])
@pytest.mark.parametrize("prompt", prompts)
@pytest.mark.skip  # will be enabled in nightly since are computationally expensive
def test_beam_search_long_sentences(model_fixture, num_beam_groups, group_size, 
                              max_new_tokens, diversity_penalty, prompt):
    generation_config = dict(
        num_beam_groups=num_beam_groups, 
        num_beams=num_beam_groups * group_size, 
        diversity_penalty=1.0, 
        num_return_sequences=num_beam_groups * group_size, 
        max_new_tokens=max_new_tokens, 
    )
    run_hf_ov_genai_comparison(model_fixture, generation_config, prompt)
