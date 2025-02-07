# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import json
import functools
import shutil
import openvino
from pathlib import Path
from typing import List, Tuple, Callable

from openvino_genai import ContinuousBatchingPipeline, LLMPipeline, SchedulerConfig, GenerationResult, GenerationConfig, DecodedResults, StreamerBase
from openvino_tokenizers import convert_tokenizer

from utils.comparation_utils import compare_generation_results
from utils.ov_tokenizer_utils import delete_rt_info
from utils.test_data import get_default_properties

TESTS_ROOT = Path(__file__).parent

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


def dict_to_scheduler_config(scheduler_params: dict = None) -> SchedulerConfig:
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
 
    cb_pipe = ContinuousBatchingPipeline(models_path, scheduler_config=scheduler_config, device='CPU', tokenizer_properties={}, properties=get_default_properties())
    output = cb_pipe.generate(prompts, generation_configs)

    del cb_pipe
    shutil.rmtree(models_path)

    return output


def run_llm_pipeline(
    models_path : Path,
    prompts: List[str],
    generation_config : GenerationConfig,
    use_cb : bool = False,
    streamer: StreamerWithResults | Callable | StreamerBase = None
) -> List[GenerationResult]:
    properties = get_default_properties()
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


@functools.lru_cache(1)
def get_continuous_batching(path):
    return LLMPipeline(path, 'CPU', scheduler_config=SchedulerConfig(), **get_default_properties())


def load_genai_pipe_with_configs(configs: List[Tuple], temp_path):
    # Load LLMPipeline where all configs are cleared.
    # remove existing jsons from previous tests
    for json_file in temp_path.glob("*.json"):
        json_file.unlink()
    delete_rt_info(configs, temp_path)

    for config_json, config_name in configs:
        with (temp_path / config_name).open('w') as f:
            json.dump(config_json, f)

    ov_pipe = LLMPipeline(temp_path, 'CPU', **get_default_properties())

    for _, config_name in configs:
        os.remove(temp_path / config_name)

    return ov_pipe
