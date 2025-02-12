# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from pathlib import Path
from typing import List, Callable
from shutil import rmtree

from openvino_genai import SchedulerConfig, draft_model, ContinuousBatchingPipeline, \
    LLMPipeline, GenerationConfig, GenerationResult, StreamerBase

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


class PipelineType(Enum):
    STATEFUL = 1
    STATELESS = 2
    CONTINIOUS_BATCHING = 3
    SPECULATIVE_DECODING = 4
    PROMPT_LOOKUP_DECODING = 5


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


def create_ov_pipeline(models_path: Path,
                       pipeline_type: PipelineType = PipelineType.STATEFUL,
                       device: str = "CPU",
                       ov_config: dict = {},
                       scheduler_config: SchedulerConfig = SchedulerConfig(),
                       draft_model: draft_model = None):
    if pipeline_type == PipelineType.STATEFUL:
        return LLMPipeline(models_path, device, ov_config)
    elif pipeline_type == PipelineType.STATELESS:
        return LLMPipeline(models_path, device, ov_config, scheduler_config=scheduler_config)
    elif pipeline_type == PipelineType.CONTINIOUS_BATCHING:
        return ContinuousBatchingPipeline(models_path, scheduler_config, device, ov_config)
    elif pipeline_type == PipelineType.SPECULATIVE_DECODING:
        return LLMPipeline(models_path, device, ov_config, scheduler_config=scheduler_config, draft_model=(draft_model(models_path) if draft_model is None else draft_model))
    elif pipeline_type == PipelineType.PROMPT_LOOKUP_DECODING:
        return LLMPipeline(models_path, device, ov_config, scheduler_config=scheduler_config, prompt_lookup=True)
    else:
        raise Exception(f"Unsupported pipeline type: {pipeline_type}")


def prepare_generation_config_by_pipe_type(generation_config : GenerationConfig,
                                           pipeline_type: PipelineType = PipelineType.STATEFUL):
    if pipeline_type == PipelineType.SPECULATIVE_DECODING:
        generation_config.assistant_confidence_threshold = 0.9
    elif pipeline_type == PipelineType.PROMPT_LOOKUP_DECODING:
        generation_config.num_assistant_tokens = 5
        generation_config.max_ngram_size = 3
    else:
        pass
    return generation_config

def prepare_generation_configs_by_pipe_type(generation_configs : List[GenerationConfig],
                                            pipeline_type: PipelineType = PipelineType.STATEFUL):
    return [ prepare_generation_config_by_pipe_type(generation_config, pipeline_type) for generation_config in generation_configs ]

# todo: typing
def run_ov_pipeline(models_path: Path,
                    prompt : str | List[str],
                    generation_config : GenerationConfig | List[GenerationConfig],
                    pipeline_type : PipelineType = PipelineType.STATEFUL,
                    streamer: StreamerWithResults | Callable | StreamerBase = None,
                    scheduler_config: SchedulerConfig = SchedulerConfig(),
                    draft_model: draft_model = None,
                    ov_config: dict = {},
                    device: str = "CPU"
    ):
    # update the generation config according pipeline_type
    updated_generation_config = None
    if isinstance(generation_config, List[GenerationConfig]):
        if pipeline_type != PipelineType.CONTINIOUS_BATCHING:
            raise Exception(f"\'generation_config\' is \'List[GenerationConfig]\'. This type is supported only for \'PipelineType.CONTINIOUS_BATCHING\'! Please change pipeline_type or generation_config type!")
        assert isinstance(prompt, List[str])
        assert len(generation_config) == len(prompt)
        updated_generation_config = prepare_generation_configs_by_pipe_type(generation_config, pipeline_type)
    else:
        updated_generation_config = prepare_generation_configs_by_pipe_type(generation_config, pipeline_type)

    ov_pipe = create_ov_pipeline(models_path=models_path,
                                 pipeline_type=pipeline_type,
                                 device=device,
                                 ov_config=ov_config,
                                 scheduler_config=scheduler_config,
                                 draft_model=draft_model)

    results = ov_pipe.generate(prompt, updated_generation_config)
    
    del ov_pipe
    rmtree(models_path)

    

    return results

