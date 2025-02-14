# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from pathlib import Path

from openvino_genai import SchedulerConfig, draft_model, ContinuousBatchingPipeline, LLMPipeline

from utils.constants import get_default_llm_properties

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


def create_ov_pipeline(models_path: Path,
                       pipeline_type: PipelineType = PipelineType.STATEFUL,
                       device: str = "CPU",
                       ov_config: dict = get_default_llm_properties(),
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

