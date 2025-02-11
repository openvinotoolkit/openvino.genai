# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from pathlib import Path

from openvino_genai import SchedulerConfig, draft_model, ContinuousBatchingPipeline, LLMPipeline


class PipelineType(Enum):
    STATEFUL = 1
    STATELESS = 2
    CONTINIOUS_BATCHING = 3
    SPECULATIVE_DECODING = 4
    PROMPT_LOOKUP_DECODING = 5


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

