# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import pytest
import openvino

from optimum.intel import OVModelForCausalLM
from pathlib import Path
from openvino_genai import ContinuousBatchingPipeline, LLMPipeline, SchedulerConfig, GenerationResult, GenerationConfig, DecodedResults, StopCriteria, StreamerBase, Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig as HFGenerationConfig
from typing import List, Tuple, Callable

from utils.generation_config import get_greedy, get_beam_search
from utils.constants import get_default_llm_properties
from utils.hugging_face import convert_models, get_hugging_face_models, run_hugging_face
from utils.comparation import compare_generation_results

TESTS_ROOT = Path(__file__).parent

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


def run_continuous_batching(
    models_path : Path,
    scheduler_config : SchedulerConfig,
    prompts: List[str],
    generation_configs : List[GenerationConfig] | GenerationConfig 
) -> List[GenerationResult]:
    if type(generation_configs) is not list:
        generation_configs = [generation_configs] * len(prompts)
 
    cb_pipe = ContinuousBatchingPipeline(models_path, scheduler_config=scheduler_config, device='CPU', tokenizer_properties={}, properties=get_default_llm_properties())
    output = cb_pipe.generate(prompts, generation_configs)

    del cb_pipe
    shutil.rmtree(models_path)

    return output


def get_models_list_from_path(file_name: str):
    models = []
    with open(file_name) as f:
        for model_name in f:
            model_name = model_name.strip()
            # skip comment in model scope file
            if model_name.startswith('#'):
                continue
            models.append(model_name)
    return models


class StreamerWithResults:
    # Return a streamer which accumulates results in order to compare with results returned from generate.
    results: List[str] = []
    def __init__(self):
        self.results = []

    def accumulate(self, subword) -> bool:
        self.results.append(subword)
        return False
    
    def get_results(self) -> List[GenerationResult]:
        streaming_result = GenerationResult()
        streaming_result.m_generation_ids = [''.join(self.results)]
        return [streaming_result]
    
    def reset(self):
        self.results = []



def run_llm_pipeline(
    models_path : Path,
    prompts: List[str],
    generation_config : GenerationConfig,
    use_cb : bool = False,
    streamer: StreamerWithResults | Callable | StreamerBase = None
) -> List[GenerationResult]:
    properties = get_default_llm_properties()
    if use_cb:
        properties['scheduler_config'] = SchedulerConfig()
    ov_pipe = LLMPipeline(models_path, device='CPU', **properties)
    
    if streamer is None and not (generation_config.is_beam_search() or generation_config.num_return_sequences > 1) and len(prompts) == 1:
        # We can use streamer only if we have a single prompt and not beam search.
        streamer = StreamerWithResults()
    if isinstance(streamer, StreamerWithResults):
        # Clear the accumulated strings to avoid side effects
        streamer.reset()

    generate_outputs : DecodedResults = ov_pipe.generate(
        inputs=prompts, 
        generation_config=generation_config, 
        streamer=streamer.accumulate if isinstance(streamer, StreamerWithResults) else streamer
    )

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
    
    if isinstance(streamer, StreamerWithResults):
        compare_generation_results(prompts, generation_results, streamer.get_results(), generation_config)

    return generation_results


def run_llm_pipeline_with_ref(model_id: str, 
                              prompts: List[str], 
                              generation_config: GenerationConfig | dict, 
                              tmp_path: Path, 
                              use_cb : bool = False,
                              streamer: StreamerWithResults | Callable | StreamerBase = None):
    models_path : Path = tmp_path / model_id
    opt_model, hf_tokenizer = get_hugging_face_models(model_id)

    if type(generation_config) is dict:
        generation_config = GenerationConfig(**generation_config)

    convert_models(opt_model, hf_tokenizer, models_path)

    ov_results = run_llm_pipeline(models_path, prompts, generation_config, use_cb, streamer=streamer.accumulate if isinstance(streamer, StreamerWithResults) else streamer)
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

"""rt_info has the highest priority. Delete it to respect configs."""
def delete_rt_info(configs: List[Tuple], temp_path):
    core = openvino.Core()
    core.set_property({'ENABLE_MMAP': False})
    for model_path in temp_path / "openvino_tokenizer.xml", temp_path / "openvino_detokenizer.xml":
        tokenizer = core.read_model(model_path)
        rt_info = tokenizer.get_rt_info()
        for config, _ in configs:
            for key in config.keys():
                # tokenizer_config.json contains strings instead of ids so the keys don't have "_id".
                for modified_key in (key, key+"_id"):
                    try:
                        del rt_info[modified_key]
                    except KeyError:
                        pass
        openvino.save_model(tokenizer, model_path)
