# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import pytest

from optimum.intel import OVModelForCausalLM
from pathlib import Path
from openvino_genai import ContinuousBatchingPipeline, LLMPipeline, SchedulerConfig, GenerationResult, GenerationConfig, DecodedResults, StopCriteria
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig as HFGenerationConfig
from typing import List, Tuple

TESTS_ROOT = Path(__file__).parent

def get_greedy() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 30
    return generation_config

def get_greedy_with_penalties() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.presence_penalty = 2.0
    generation_config.frequency_penalty = 0.2
    generation_config.max_new_tokens = 30
    return generation_config

def get_beam_search() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_beam_groups = 3
    generation_config.num_beams = 6
    generation_config.diversity_penalty = 1
    generation_config.max_new_tokens = 30
    generation_config.num_return_sequences = 3
    generation_config.num_return_sequences = generation_config.num_beams
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
    if generation_config is None:
        return

    kwargs = {}
    kwargs['return_dict_in_generate'] = True

    # generic parameters
    kwargs['max_length'] = generation_config.max_length
    # has higher priority than 'max_length'
    kwargs['max_new_tokens'] = generation_config.max_new_tokens
    kwargs['min_new_tokens'] = generation_config.min_new_tokens
    if generation_config.stop_strings:
        kwargs['stop_strings'] = generation_config.stop_strings

    # copy default parameters
    kwargs['bos_token_id'] = default_generation_config.bos_token_id
    kwargs['pad_token_id'] = default_generation_config.pad_token_id

    if len(generation_config.stop_token_ids) > 0:
        kwargs['eos_token_id'] = list(generation_config.stop_token_ids)
    elif generation_config.eos_token_id != -1:
        kwargs['eos_token_id'] = generation_config.eos_token_id
    else:
        kwargs['eos_token_id'] = default_generation_config.eos_token_id

    # copy penalties
    kwargs['repetition_penalty'] = generation_config.repetition_penalty

    if generation_config.is_beam_search():
        # beam search case
        kwargs['num_beam_groups'] = generation_config.num_beam_groups
        kwargs['num_beams'] = generation_config.num_beams
        kwargs['length_penalty'] = generation_config.length_penalty
        kwargs['no_repeat_ngram_size'] = generation_config.no_repeat_ngram_size
        kwargs['num_return_sequences'] = generation_config.num_return_sequences
        kwargs['output_scores'] = True

        if generation_config.num_beam_groups > 1:
            kwargs['diversity_penalty'] = generation_config.diversity_penalty

        # in OpenVINO GenAI this parameter is called stop_criteria,
        # while in HF it's called early_stopping.
        # HF values True, False and "never" correspond to OV GenAI values "EARLY", "HEURISTIC" and "NEVER"
        STOP_CRITERIA_MAP = {
            StopCriteria.NEVER: "never",
            StopCriteria.EARLY: True,
            StopCriteria.HEURISTIC: False
        }

        kwargs['early_stopping'] = STOP_CRITERIA_MAP[generation_config.stop_criteria]
    elif generation_config.is_multinomial():
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
    generation_configs: List[GenerationConfig] | GenerationConfig,
) -> List[GenerationResult]:
    generation_results = []

    if type(generation_configs) is list:
        # process prompt by promp as we have multiple generation configs
        for prompt, generation_config in zip(prompts, generation_configs):
            hf_generation_config = convert_to_hf(opt_model.generation_config, generation_config)
            inputs = hf_tokenizer(prompt, return_tensors="pt")
            input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
            prompt_len = 0 if generation_config.echo else input_ids.numel()

            generate_outputs = opt_model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=hf_generation_config, tokenizer=hf_tokenizer)
            all_text_batch = hf_tokenizer.batch_decode([generated_ids[prompt_len:] for generated_ids in generate_outputs.sequences], skip_special_tokens=True)

            generation_result = GenerationResult()
            generation_result.m_generation_ids = all_text_batch
            # sequences_scores are available only for beam search case
            if generation_config.is_beam_search():
                generation_result.m_scores = [score for score in generate_outputs.sequences_scores]
            generation_results.append(generation_result)
    else:
        # process all prompts as a single batch as we have a single generation config for all prompts
        inputs = hf_tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True, padding_side='left')
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        hf_generation_config = convert_to_hf(opt_model.generation_config, generation_configs)
        hf_encoded_outputs = opt_model.generate(input_ids, attention_mask=attention_mask, generation_config=hf_generation_config, tokenizer=hf_tokenizer)

        generation_ids = []
        scores = []

        for idx, hf_encoded_out in enumerate(hf_encoded_outputs.sequences):
            prompt_idx = idx // hf_generation_config.num_return_sequences
            prompt_len = 0 if generation_configs.echo else input_ids[prompt_idx].numel()
            decoded_text = hf_tokenizer.decode(hf_encoded_out[prompt_len:], skip_special_tokens=True)
            generation_ids.append(decoded_text)
            if generation_configs.is_beam_search():
                scores.append(hf_encoded_outputs.sequences_scores[idx])

            # if we need to move to next generation result
            if (idx + 1) // hf_generation_config.num_return_sequences != prompt_idx:
                generation_result = GenerationResult()
                generation_result.m_generation_ids = generation_ids
                generation_result.m_scores = scores
                generation_results.append(generation_result)
                generation_ids = []
                scores = []

    del hf_tokenizer
    del opt_model

    return generation_results


def run_continuous_batching(
    models_path : Path,
    scheduler_config : SchedulerConfig,
    prompts: List[str],
    generation_configs : List[GenerationConfig] | GenerationConfig 
) -> List[GenerationResult]:
    if type(generation_configs) is not list:
        generation_configs = [generation_configs] * len(prompts)
 
    cb_pipe = ContinuousBatchingPipeline(models_path, scheduler_config=scheduler_config, device='CPU', tokenizer_properties={}, properties=get_default_properties())
    output = cb_pipe.generate(prompts, generation_configs)

    del cb_pipe
    shutil.rmtree(models_path)

    return output


def get_default_properties():
    import openvino.properties.hint as hints
    import openvino as ov

    return {
        hints.inference_precision : ov.Type.f32,
        hints.kv_cache_precision : ov.Type.f16,
    }


def run_llm_pipeline(
    models_path : Path,
    prompts: List[str],
    generation_config : GenerationConfig,
    use_cb : bool = False
) -> List[GenerationResult]:
    properties = get_default_properties()
    if use_cb:
        properties['scheduler_config'] = SchedulerConfig()

    ov_pipe = LLMPipeline(models_path, device='CPU', **properties)

    generate_outputs : DecodedResults = ov_pipe.generate(inputs=prompts, generation_config=generation_config)

    index = 0
    generation_results = []

    for _ in prompts:
        generation_result = GenerationResult()

        generation_result.m_generation_ids = generate_outputs.texts[index : index + generation_config.num_return_sequences]
        # sequences_scores are available only for beam search case
        if generation_config.is_beam_search():
            generation_result.m_scores = generate_outputs.scores[index : index + generation_config.num_return_sequences]
        generation_results.append(generation_result)

        index += generation_config.num_return_sequences

    del ov_pipe
    shutil.rmtree(models_path)

    return generation_results


def compare_generation_result(hf_result: GenerationResult, ov_result: GenerationResult, generation_config: GenerationConfig):
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


def compare_generation_results(prompts: List[str], hf_results: List[GenerationResult], ov_results: List[GenerationResult], generation_configs: List[GenerationConfig] | GenerationConfig):
    if type(generation_configs) is not list:
        generation_configs = [generation_configs]

    assert len(prompts) == len(hf_results)
    assert len(prompts) == len(ov_results)

    for prompt, ref_result, ov_result, generation_config in zip(prompts, hf_results, ov_results, generation_configs):
        print(f"Prompt = {prompt}\nReference result = {ref_result}\nOpenVINO result = {ov_result.m_generation_ids}")
        compare_generation_result(ref_result, ov_result, generation_config)


def get_hugging_face_models(model_id: str):
    hf_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    opt_model = OVModelForCausalLM.from_pretrained(model_id, export=True, compile=False, load_in_8bit=False, trust_remote_code=True, ov_config=get_default_properties())
    return opt_model, hf_tokenizer


def convert_models(opt_model : OVModelForCausalLM, hf_tokenizer : AutoTokenizer, models_path: Path):
    opt_model.save_pretrained(models_path)

    # to store tokenizer config jsons with special tokens
    hf_tokenizer.save_pretrained(models_path)

    # save generation config
    opt_model.generation_config.save_pretrained(models_path)

    # convert tokenizers as well
    from openvino_tokenizers import convert_tokenizer
    from openvino import serialize

    tokenizer, detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    serialize(tokenizer, models_path / "openvino_tokenizer.xml")
    serialize(detokenizer, models_path / "openvino_detokenizer.xml")


def run_llm_pipeline_with_ref(model_id: str, prompts: List[str], generation_config: GenerationConfig | dict, tmp_path: Path, use_cb : bool = False):
    models_path : Path = tmp_path / model_id
    opt_model, hf_tokenizer = get_hugging_face_models(model_id)

    if type(generation_config) is dict:
        generation_config = GenerationConfig(**generation_config)

    convert_models(opt_model, hf_tokenizer, models_path)

    ov_results = run_llm_pipeline(models_path, prompts, generation_config, use_cb)
    hf_results = run_hugging_face(opt_model, hf_tokenizer, prompts, generation_config)

    compare_generation_results(prompts, hf_results, ov_results, generation_config)


def run_cb_pipeline_with_ref(tmp_path: str, model_id: str, scheduler_params: dict = {}, generation_config : GenerationConfig | dict = None):
    prompts, generation_configs = get_test_dataset()
    scheduler_config = get_scheduler_config(scheduler_params)

    # override dataset's generation config
    if generation_config is not None:
        if type(generation_config) is dict:
            generation_config = GenerationConfig(**generation_config)
        generation_configs = [generation_config] * len(prompts)

    models_path : Path = tmp_path / model_id
    opt_model, hf_tokenizer = get_hugging_face_models(model_id)

    convert_models(opt_model, hf_tokenizer, models_path)

    hf_results = run_hugging_face(opt_model, hf_tokenizer, prompts, generation_configs)
    ov_results = run_continuous_batching(models_path, scheduler_config, prompts, generation_configs)

    compare_generation_results(prompts, hf_results, ov_results, generation_configs)


# TODO: remove after Generator property is supported by LLMPipeline / VLMPipeline
def generate_and_compare_with_reference_text(models_path: Path, prompts: List[str], reference_texts_per_prompt: List[List[str]], generation_configs: List[GenerationConfig], scheduler_config: SchedulerConfig):
    ov_results : List[GenerationResult] = run_continuous_batching(models_path, scheduler_config, prompts, generation_configs)

    assert len(prompts) == len(reference_texts_per_prompt)
    assert len(prompts) == len(ov_results)

    for prompt, ref_texts_for_this_prompt, ov_result in zip(prompts, reference_texts_per_prompt, ov_results):
        print(f"Prompt = {prompt}\nref text = {ref_texts_for_this_prompt}\nOV result = {ov_result.m_generation_ids}")

        assert len(ref_texts_for_this_prompt) == len(ov_result.m_generation_ids)
        for ref_text, ov_text in zip(ref_texts_for_this_prompt, ov_result.m_generation_ids):
            assert ref_text == ov_text


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

def get_streamer_with_results():
    # Return a streamer which accumulates results in order to compare with results returned from generate.
    class StreamerWithResults:
        results: List[str] = []
        def __init__(self):
            self.results = []

        def accumulate(self, subword) -> bool:
            self.results.append(subword)
            return False
        
        def get_result_str(self) -> str:
            return ''.join(self.results)
        
        def reset(self):
            self.results = []

    return StreamerWithResults()
