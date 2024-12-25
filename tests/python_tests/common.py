# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import pytest

from optimum.intel import OVModelForCausalLM
from pathlib import Path
from openvino_genai import ContinuousBatchingPipeline, SchedulerConfig, GenerationResult, GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig as HFGenerationConfig
from typing import List, Tuple

TESTS_ROOT = Path(__file__).parent

def get_greedy() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 30
    return generation_config

def get_greedy_with_min_and_max_tokens() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.min_new_tokens = 15
    generation_config.max_new_tokens = 30
    return generation_config

def get_greedy_with_repetition_penalty() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.repetition_penalty = 2.0
    generation_config.max_new_tokens = 30
    return generation_config

def get_greedy_with_penalties() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.presence_penalty = 2.0
    generation_config.frequency_penalty = 0.2
    generation_config.max_new_tokens = 30
    return generation_config

def get_greedy_with_single_stop_string() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.min_new_tokens = 15
    generation_config.max_new_tokens = 50
    generation_config.stop_strings = {"anag"} # expected match on "manage"
    generation_config.include_stop_str_in_output = True
    return generation_config

def get_greedy_with_multiple_stop_strings() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.min_new_tokens = 1
    generation_config.max_new_tokens = 50
    generation_config.stop_strings = {".", "software", "Intel"}
    generation_config.include_stop_str_in_output = True
    return generation_config

def get_greedy_with_multiple_stop_strings_no_match() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.min_new_tokens = 1
    generation_config.max_new_tokens = 50
    generation_config.stop_strings = {"Einstein", "sunny", "geothermal"}
    generation_config.include_stop_str_in_output = True
    return generation_config

def get_beam_search() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_beam_groups = 3
    generation_config.num_beams = 6
    generation_config.max_new_tokens = 30
    generation_config.num_return_sequences = 3
    generation_config.num_return_sequences = generation_config.num_beams
    return generation_config

def get_beam_search_min_and_max_tokens() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_beam_groups = 3
    generation_config.num_beams = 6
    generation_config.min_new_tokens = 15
    generation_config.max_new_tokens = 30
    generation_config.num_return_sequences = 3
    generation_config.num_return_sequences = generation_config.num_beams
    return generation_config

def get_beam_search_with_single_stop_string() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_beam_groups = 3
    generation_config.num_beams = 6
    generation_config.max_new_tokens = 50
    generation_config.num_return_sequences = generation_config.num_beams
    generation_config.stop_strings = {"open sour"}  # expected match on "open source"
    generation_config.include_stop_str_in_output = True
    return generation_config

def get_beam_search_with_multiple_stop_strings() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_beam_groups = 3
    generation_config.num_beams = 6
    generation_config.max_new_tokens = 50
    generation_config.num_return_sequences = generation_config.num_beams
    generation_config.stop_strings = {".", "software", "Intel"}
    generation_config.include_stop_str_in_output = True
    return generation_config

def get_beam_search_with_multiple_stop_strings_no_match() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_beam_groups = 3
    generation_config.num_beams = 6
    generation_config.max_new_tokens = 30
    generation_config.num_return_sequences = generation_config.num_beams
    generation_config.stop_strings = {"Einstein", "sunny", "geothermal"}
    generation_config.include_stop_str_in_output = True
    return generation_config

def get_greedy_stop_strings_exclude_from_output() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.max_new_tokens = 30
    generation_config.stop_strings = { "machines" }
    generation_config.include_stop_str_in_output = False
    return generation_config

def get_greedy_stop_strings_include_to_output() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.max_new_tokens = 30
    generation_config.stop_strings = { "machines" }
    generation_config.include_stop_str_in_output = True
    return generation_config

def get_greedy_n_stop_strings_exclude_from_output() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.max_new_tokens = 30
    generation_config.stop_strings = { "machines", "manage" }
    generation_config.include_stop_str_in_output = False
    return generation_config

def get_greedy_n_stop_strings_include_to_output() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.max_new_tokens = 30
    generation_config.stop_strings = { "machines", "manage" }
    generation_config.include_stop_str_in_output = True
    return generation_config

def get_multinomial_temperature() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.temperature = 0.8
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_temperature_and_num_return_sequence() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.temperature = 0.7
    generation_config.num_return_sequences = 3
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_temperature_and_top_p() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.do_sample = True
    generation_config.temperature = 0.8
    generation_config.top_p = 0.9
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_temperature_and_top_k() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.num_return_sequences = 1
    generation_config.temperature = 0.8
    generation_config.top_k = 2
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_temperature_top_p_and_top_k() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.temperature = 0.8
    generation_config.top_p = 0.9
    generation_config.num_return_sequences = 1
    generation_config.top_k = 2
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_temperature_and_repetition_penalty() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.num_return_sequences = 1
    generation_config.temperature = 0.8
    generation_config.repetition_penalty = 2.0
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_all_parameters() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.num_return_sequences = 4
    generation_config.temperature = 0.9
    generation_config.top_p = 0.8
    generation_config.top_k = 20
    generation_config.repetition_penalty = 2.0
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_temperature_and_frequence_penalty() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.temperature = 0.8
    generation_config.frequency_penalty = 0.5
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_temperature_and_presence_penalty() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.temperature = 0.8
    generation_config.presence_penalty = 0.1
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_max_and_min_token() -> GenerationConfig:
    multinomial = GenerationConfig()
    multinomial.do_sample = True
    multinomial.temperature = 0.9
    multinomial.top_p = 0.9
    multinomial.top_k = 20
    multinomial.num_return_sequences = 3
    multinomial.presence_penalty = 0.01
    multinomial.frequency_penalty = 0.1
    multinomial.min_new_tokens = 15
    multinomial.max_new_tokens = 30
    return multinomial

def get_test_dataset() -> Tuple[List[str], List[GenerationConfig]]:
    prompts = [
        "What is OpenVINO?",
        "How are you?",
        "What is your name?",
        "Tell me something about Canada"
    ]
    generation_configs = [
        get_greedy(),
        get_beam_search(),
        get_greedy(),
        get_beam_search(),
    ]
    return (prompts, generation_configs)


def get_scheduler_config(scheduler_params: dict = None) -> SchedulerConfig:
    scheduler_config = SchedulerConfig()
    if scheduler_params is None:
        scheduler_config.dynamic_split_fuse = True
        # vLLM specific
        scheduler_config.max_num_batched_tokens = 256
        scheduler_config.max_num_seqs = 256

        # Expedited number of blocks = text_blocks_n * G * n_prompts, where
        # text_blocks_n - number of blocks required for storing prompt and generated text,
        # currently it is 1 block for prompt (31 token with block_size 32) + 1 block for generated text (max length of generated text - 30 tokens);
        # G - number of sequences in a sequence group, for beam search it is 2(group_size) * 3 (num_groups);
        # n_prompts - number of prompts.
        # For current parameters in tests expedited number of blocks is approximately 48.
        scheduler_config.num_kv_blocks = 60
    else:
        for param, value in scheduler_params.items():
            setattr(scheduler_config, param, value)

    return scheduler_config


def convert_to_hf(
    default_generation_config : HFGenerationConfig,
    generation_config : GenerationConfig
) -> HFGenerationConfig:
    kwargs = {}

    # generic parameters
    kwargs['max_length'] = generation_config.max_length
    # has higher priority than 'max_length'
    kwargs['max_new_tokens'] = generation_config.max_new_tokens
    kwargs['min_new_tokens'] = generation_config.min_new_tokens
    if generation_config.stop_strings:
        kwargs['stop_strings'] = generation_config.stop_strings

    # copy default parameters
    kwargs['bos_token_id'] = default_generation_config.bos_token_id
    kwargs['eos_token_id'] = default_generation_config.eos_token_id
    kwargs['pad_token_id'] = default_generation_config.pad_token_id
    kwargs['repetition_penalty'] = generation_config.repetition_penalty

    if generation_config.num_beams > 1:
        # beam search case
        kwargs['num_beam_groups'] = generation_config.num_beam_groups
        kwargs['num_beams'] = generation_config.num_beams
        kwargs['length_penalty'] = generation_config.length_penalty
        kwargs['no_repeat_ngram_size'] = generation_config.no_repeat_ngram_size
        kwargs['num_return_sequences'] = generation_config.num_return_sequences
        kwargs['output_scores'] = True
        if generation_config.num_beam_groups > 1:
            kwargs['diversity_penalty'] = generation_config.diversity_penalty
    elif generation_config.do_sample:
        # mulitinomial
        kwargs['temperature'] = generation_config.temperature
        kwargs['top_k'] = generation_config.top_k
        kwargs['top_p'] = generation_config.top_p
        kwargs['do_sample'] = generation_config.do_sample
    else:
        # greedy
        pass

    hf_generation_config = HFGenerationConfig(**kwargs)
    return hf_generation_config


def run_hugging_face(
    opt_model,
    hf_tokenizer,
    prompts: List[str],
    generation_configs: List[GenerationConfig],
) -> List[GenerationResult]:
    generation_results = []
    for prompt, generation_config in zip(prompts, generation_configs):
        inputs = hf_tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs['input_ids'].numel()
        generate_outputs = opt_model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                              generation_config=convert_to_hf(opt_model.generation_config, generation_config),
                                              return_dict_in_generate=True, tokenizer=hf_tokenizer)
        all_text_batch = hf_tokenizer.batch_decode([generated_ids[prompt_len:] for generated_ids in generate_outputs.sequences], skip_special_tokens=True)

        generation_result = GenerationResult()
        generation_result.m_generation_ids = all_text_batch
        # sequences_scores are available only for beam search case
        if generation_config.is_beam_search():
            generation_result.m_scores = [score for score in generate_outputs.sequences_scores]
        generation_results.append(generation_result)

    del hf_tokenizer
    del opt_model

    return generation_results


def run_continuous_batching(
    models_path : Path,
    scheduler_config : SchedulerConfig,
    prompts: List[str],
    generation_configs : List[GenerationConfig]
) -> List[GenerationResult]:
    pipe = ContinuousBatchingPipeline(models_path, scheduler_config, "CPU")
    output = pipe.generate(prompts, generation_configs)
    del pipe
    shutil.rmtree(models_path)
    return output


def read_models_list(file_name: str):
    models = []
    with open(file_name) as f:
        for model_name in f:
            model_name = model_name.strip()
            # skip comment in model scope file
            if model_name.startswith('#'):
                continue
            models.append(model_name)
    return models


def compare_results(hf_result: GenerationResult, ov_result: GenerationResult, generation_config: GenerationConfig):
    if generation_config.is_beam_search():
        assert len(hf_result.m_scores) == len(ov_result.m_scores)
        for hf_score, ov_score in zip(hf_result.m_scores, ov_result.m_scores):
            # Note, that for fp32 / fp16 models scores are different less than 0.001
            assert abs(hf_score - ov_score) < 0.02

    if not generation_config.include_stop_str_in_output and len(generation_config.stop_strings) > 0:
        assert len(hf_result.m_generation_ids) >= len(ov_result.m_generation_ids)
        for hf_text, ov_text in zip(hf_result.m_generation_ids, ov_result.m_generation_ids):
            assert ov_text in hf_text
    else:
        assert len(hf_result.m_generation_ids) == len(ov_result.m_generation_ids)
        for hf_text, ov_text in zip(hf_result.m_generation_ids, ov_result.m_generation_ids):
            assert hf_text == ov_text


def get_hugging_face_model_and_tokenizer(model_id: str, use_optimum = True):
    hf_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    opt_model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True) if use_optimum else \
                AutoModelForCausalLM.from_pretrained(model_id)
    return opt_model, hf_tokenizer


def save_ov_model_from_optimum(model, hf_tokenizer, models_path: Path):
    model.save_pretrained(models_path)
    # convert tokenizers as well
    from openvino_tokenizers import convert_tokenizer
    from openvino import serialize
    tokenizer, detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True, skip_special_tokens=True)
    serialize(tokenizer, models_path / "openvino_tokenizer.xml")
    serialize(detokenizer, models_path / "openvino_detokenizer.xml")


def _generate_and_compare_with_reference_results(models_path: Path, prompts: List[str], reference_results: List[GenerationResult], generation_configs: List[GenerationConfig], scheduler_config: SchedulerConfig):
    ov_results : List[GenerationResult] = run_continuous_batching(models_path, scheduler_config, prompts, generation_configs)

    assert len(prompts) == len(reference_results)
    assert len(prompts) == len(ov_results)

    for prompt, ref_result, ov_result, generation_config in zip(prompts, reference_results, ov_results, generation_configs):
        print(f"Prompt = {prompt}\nref result = {ref_result}\nOV result = {ov_result.m_generation_ids}")
        compare_results(ref_result, ov_result, generation_config)


def generate_and_compare_with_hf(model_id: str, prompts: List[str], generation_configs: List[GenerationConfig], scheduler_config: SchedulerConfig, tmp_path: Path):
    use_optimum = True
    models_path : Path = tmp_path / model_id
    opt_model, hf_tokenizer = get_hugging_face_model_and_tokenizer(model_id, use_optimum)

    if use_optimum:
        save_ov_model_from_optimum(opt_model, hf_tokenizer, models_path)

    hf_results = run_hugging_face(opt_model=opt_model, hf_tokenizer=hf_tokenizer, prompts=prompts, generation_configs=generation_configs)
    _generate_and_compare_with_reference_results(models_path, prompts, hf_results, generation_configs, scheduler_config)


def generate_and_compare_with_reference_text(models_path: Path, prompts: List[str], reference_texts_per_prompt: List[List[str]], generation_configs: List[GenerationConfig], scheduler_config: SchedulerConfig):
    ov_results : List[GenerationResult] = run_continuous_batching(models_path, scheduler_config, prompts, generation_configs)

    assert len(prompts) == len(reference_texts_per_prompt)
    assert len(prompts) == len(ov_results)

    for prompt, ref_texts_for_this_prompt, ov_result in zip(prompts, reference_texts_per_prompt, ov_results):
        print(f"Prompt = {prompt}\nref text = {ref_texts_for_this_prompt}\nOV result = {ov_result.m_generation_ids}")

        assert len(ref_texts_for_this_prompt) == len(ov_result.m_generation_ids)
        for ref_text, ov_text in zip(ref_texts_for_this_prompt, ov_result.m_generation_ids):
            assert ref_text == ov_text


def run_test_pipeline(tmp_path: str, model_id: str, scheduler_params: dict = None, generation_config = None):
    prompts, generation_configs = get_test_dataset()
    scheduler_config = get_scheduler_config(scheduler_params)

    if generation_config is not None:
        generation_config.rng_seed = 0
        generation_configs = [generation_config] * len(prompts)

    generate_and_compare_with_hf(model_id, prompts, generation_configs, scheduler_config, tmp_path)


DEFAULT_SCHEDULER_CONFIG = get_scheduler_config({"num_kv_blocks": 300, "dynamic_split_fuse": True, "max_num_batched_tokens": 256, "max_num_seqs": 256})

def get_image_by_link(link):
    from PIL import Image
    import requests
    from openvino import Tensor
    import numpy as np

    image = Image.open(requests.get(link, stream=True).raw)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_data = np.array((np.array(image.getdata()) - 128).astype(np.byte)).reshape(1, 3, image.size[1], image.size[0])
    return Tensor(image_data)
