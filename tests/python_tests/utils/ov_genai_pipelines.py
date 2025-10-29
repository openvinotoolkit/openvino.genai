# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from pathlib import Path
from typing import Callable
from shutil import rmtree

from optimum.intel.openvino.utils import TemporaryDirectory
from openvino_genai import SchedulerConfig, draft_model, ContinuousBatchingPipeline, \
    LLMPipeline, GenerationConfig, GenerationResult, StreamerBase, DecodedResults

from utils.constants import get_default_llm_properties
from utils.comparation import compare_generation_results, compare_generation_results_vs_ref
from utils.hugging_face import download_and_convert_model, run_hugging_face

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
    PAGED_ATTENTION = 2
    CONTINUOUS_BATCHING = 3
    SPECULATIVE_DECODING = 4
    PROMPT_LOOKUP_DECODING = 5
    AUTO = 6


def get_all_pipeline_types():
    return [PipelineType.STATEFUL, PipelineType.PAGED_ATTENTION, PipelineType.CONTINUOUS_BATCHING, PipelineType.SPECULATIVE_DECODING, PipelineType.PROMPT_LOOKUP_DECODING, PipelineType.AUTO]

def get_main_pipeline_types():
    return [PipelineType.STATEFUL, PipelineType.PAGED_ATTENTION, PipelineType.SPECULATIVE_DECODING, PipelineType.PROMPT_LOOKUP_DECODING]

def get_gguf_pipeline_types():
    return [PipelineType.STATEFUL, PipelineType.PAGED_ATTENTION]

class StreamerWithResults:
    # Return a streamer which accumulates results in order to compare with results returned from generate.
    results: list[str] = []
    def __init__(self):
        self.results = []

    def accumulate(self, subword) -> bool:
        self.results.append(subword)
        return False
    
    def get_results(self) -> list[GenerationResult]:
        streaming_result = GenerationResult()
        streaming_result.m_generation_ids = [''.join(self.results)]
        return [streaming_result]
    
    def reset(self):
        self.results = []


def create_ov_pipeline(models_path: Path,
                       pipeline_type: PipelineType = PipelineType.AUTO,
                       device: str = "CPU",
                       ov_config: dict = get_default_llm_properties(),
                       scheduler_config: SchedulerConfig = SchedulerConfig(),
                       draft_model_path: Path = None,
                       enable_save_ov_model: bool = None,
                       dynamic_quantization_group_size: str = None):
    local_ov_config = ov_config.copy()
    if pipeline_type == PipelineType.AUTO:
        return LLMPipeline(models_path, device, ov_config)
    elif pipeline_type == PipelineType.STATEFUL:
        if enable_save_ov_model is not None: local_ov_config["enable_save_ov_model"] = enable_save_ov_model
        if dynamic_quantization_group_size is not None: local_ov_config["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = dynamic_quantization_group_size
        return LLMPipeline(models_path, device, local_ov_config, ATTENTION_BACKEND="SDPA")
    elif pipeline_type == PipelineType.PAGED_ATTENTION:
        if enable_save_ov_model is not None: local_ov_config["enable_save_ov_model"] = enable_save_ov_model
        if dynamic_quantization_group_size is not None: local_ov_config["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = dynamic_quantization_group_size
        return LLMPipeline(models_path, device, local_ov_config, scheduler_config=scheduler_config, ATTENTION_BACKEND="PA")
    elif pipeline_type == PipelineType.CONTINUOUS_BATCHING:
        return ContinuousBatchingPipeline(models_path, scheduler_config, device, ov_config)
    elif pipeline_type == PipelineType.SPECULATIVE_DECODING:
        ov_draft_model = draft_model(models_path) if draft_model_path is None else draft_model(draft_model_path)
        return LLMPipeline(models_path, device, ov_config, scheduler_config=scheduler_config, draft_model=ov_draft_model)
    elif pipeline_type == PipelineType.PROMPT_LOOKUP_DECODING:
        return LLMPipeline(models_path, device, ov_config, scheduler_config=scheduler_config, prompt_lookup=True)
    else:
        raise Exception(f"Unsupported pipeline type: {pipeline_type}")

def create_ov_cb_pipeline(models_path: Path,
                       pipeline_type: PipelineType = PipelineType.AUTO,
                       device: str = "CPU",
                       ov_config: dict = get_default_llm_properties(),
                       scheduler_config: SchedulerConfig = SchedulerConfig(),
                       draft_model_path: Path = None):
    local_ov_config = ov_config.copy()
    if pipeline_type == PipelineType.CONTINUOUS_BATCHING:
        return ContinuousBatchingPipeline(models_path, scheduler_config, device, local_ov_config)
    elif pipeline_type == PipelineType.SPECULATIVE_DECODING:
        ov_draft_model = draft_model(models_path) if draft_model_path is None else draft_model(draft_model_path)
        local_ov_config["draft_model"] = ov_draft_model
        return ContinuousBatchingPipeline(models_path, scheduler_config, device, local_ov_config)
    elif pipeline_type == PipelineType.PROMPT_LOOKUP_DECODING:
        local_ov_config["prompt_lookup"] = True
        return ContinuousBatchingPipeline(models_path, scheduler_config, device, local_ov_config)
    else:
        raise Exception(f"Unsupported pipeline type: {pipeline_type}")


def prepare_generation_config_by_pipe_type(generation_config : GenerationConfig,
                                           pipeline_type: PipelineType = PipelineType.AUTO):
    if pipeline_type == PipelineType.SPECULATIVE_DECODING:
        assert not generation_config.is_beam_search()
        generation_config.assistant_confidence_threshold = 0.9
    elif pipeline_type == PipelineType.PROMPT_LOOKUP_DECODING:
        assert not generation_config.is_beam_search()
        generation_config.num_assistant_tokens = 5
        generation_config.max_ngram_size = 3
    return generation_config


def prepare_generation_configs_by_pipe_type(generation_configs : list[GenerationConfig],
                                            pipeline_type: PipelineType = PipelineType.AUTO):
    return [ prepare_generation_config_by_pipe_type(generation_config, pipeline_type) for generation_config in generation_configs ]


def convert_decoded_results_to_generation_result(generate_outputs: DecodedResults,
                                                 num_prompts: int,
                                                 num_return_sequences: int,
                                                 is_beam_search: bool) -> list[GenerationResult]:
    index = 0
    generation_results = []

    for _ in range(num_prompts):
        generation_result = GenerationResult()

        generation_result.m_generation_ids = generate_outputs.texts[index : index + num_return_sequences]
        # sequences_scores are available only for beam search case
        if is_beam_search:
            generation_result.m_scores = generate_outputs.scores[index : index + num_return_sequences]
        generation_results.append(generation_result)

        index += num_return_sequences
    return generation_results


def run_ov_pipeline(models_path: Path,
                    prompt : str | list[str],
                    generation_config : GenerationConfig | list[GenerationConfig],
                    pipeline_type : PipelineType = PipelineType.AUTO,
                    streamer: StreamerWithResults | Callable | StreamerBase = None,
                    scheduler_config: SchedulerConfig = SchedulerConfig(),
                    draft_model_path: Path = None,
                    ov_config: dict = {},
                    device: str = "CPU"
    ) -> list[GenerationResult]:
    # update the generation config according pipeline_type
    updated_generation_config = None
    if isinstance(generation_config, list):
        if pipeline_type != PipelineType.CONTINUOUS_BATCHING:
            raise Exception(f"\'generation_config\' is \'list[GenerationConfig]\'. This type is supported only for \'PipelineType.CONTINIOUS_BATCHING\'! Please change pipeline_type or generation_config type!")
        assert isinstance(prompt, list)
        assert len(generation_config) == len(prompt)
        updated_generation_config = prepare_generation_configs_by_pipe_type(generation_config, pipeline_type)
    else:
        updated_generation_config = prepare_generation_config_by_pipe_type(generation_config, pipeline_type)

    # checking streamer
    if isinstance(prompt, str):
        if streamer is None and not (generation_config.is_beam_search() or generation_config.num_return_sequences > 1) and len(prompt) == 1:
            # We can use streamer only if we have a single prompt and not beam search.
            streamer = StreamerWithResults()
        if isinstance(streamer, StreamerWithResults):
            # Clear the accumulated strings to avoid side effects
            streamer.reset()

    # create pipeline and generate results
    ov_pipe = create_ov_pipeline(models_path=models_path,
                                 pipeline_type=pipeline_type,
                                 device=device,
                                 ov_config=ov_config,
                                 scheduler_config=scheduler_config,
                                 draft_model_path=draft_model_path)
    generation_results = ov_pipe.generate(prompt, updated_generation_config, streamer)

    # convert results to `list[GenerationResult]`
    if isinstance(generation_results, DecodedResults):
        assert isinstance(generation_config, GenerationConfig)
        num_prompts = 1 if isinstance(prompt, str) else len(prompt)
        generation_results = convert_decoded_results_to_generation_result(generation_results, num_prompts, generation_config.num_return_sequences, generation_config.is_beam_search())
    
    # cleanup test artifacts
    del ov_pipe

    # compare streaming results with generated results
    if isinstance(streamer, StreamerWithResults):
        prompts = [ prompt ] if isinstance(prompt, str) else prompt
        compare_generation_results(prompts, generation_results, streamer.get_results(), generation_config)

    return generation_results


def is_generation_available(generation_config: GenerationConfig | list[GenerationConfig],
                            pipeline_type: PipelineType):
    if type(generation_config) is GenerationConfig:
        if generation_config.is_beam_search():
            if pipeline_type == PipelineType.PROMPT_LOOKUP_DECODING or pipeline_type == PipelineType.SPECULATIVE_DECODING:
                return False
    else:
        for gen_config in generation_config:
            if gen_config.is_beam_search():
                if pipeline_type == PipelineType.PROMPT_LOOKUP_DECODING or pipeline_type == PipelineType.SPECULATIVE_DECODING:
                    return False
    return True


# TODO: remove `ref` after Generator property is supported by LLMPipeline / VLMPipeline
def generate_and_compare(model: str,
                         prompts : str | list[str],
                         generation_config: list[GenerationConfig] | GenerationConfig | dict,
                         pipeline_type: PipelineType = PipelineType.AUTO,
                         scheduler_config: SchedulerConfig | dict = SchedulerConfig(),
                         ref : list[list[str]] = None,
                         streamer: StreamerWithResults | Callable | StreamerBase = None):
    ov_prompts = prompts if type(prompts) is list else [prompts]

    ov_gen_config = GenerationConfig(**generation_config) if type(generation_config) is dict else generation_config
    hf_gen_config = ov_gen_config

    if not is_generation_available(ov_gen_config, pipeline_type):
        return

    if type(ov_gen_config) is list:
        assert len(ov_gen_config) == len(ov_prompts)
    elif pipeline_type == PipelineType.CONTINUOUS_BATCHING:
        ov_gen_config = [ov_gen_config] * len(ov_prompts)

    ov_scheduler_config = scheduler_config if isinstance(scheduler_config, SchedulerConfig) else dict_to_scheduler_config(scheduler_config)
    opt_model, hf_tokenizer, models_path = download_and_convert_model(model)

    # w/a to align different API between CB and LLM
    run_cnt = len(ov_gen_config) if pipeline_type != PipelineType.CONTINUOUS_BATCHING and type(ov_gen_config) is list else 1

    for i in range(run_cnt):
        current_it_prompts = [ov_prompts[i]] if run_cnt > 1 else ov_prompts
        current_it_gen_config = ov_gen_config[i] if run_cnt > 1 else ov_gen_config

        ov_results = run_ov_pipeline(models_path=models_path,
                                     prompt=current_it_prompts,
                                     generation_config=current_it_gen_config,
                                     pipeline_type=pipeline_type,
                                     streamer=streamer.accumulate if isinstance(streamer, StreamerWithResults) else streamer,
                                     scheduler_config=ov_scheduler_config,
                                     ov_config=get_default_llm_properties())

        if ref is None:
            current_it_hf_config = [hf_gen_config[i]] if run_cnt > 1 else hf_gen_config
            ref_results = run_hugging_face(opt_model, hf_tokenizer, current_it_prompts, current_it_hf_config)
            compare_generation_results(current_it_prompts, ref_results, ov_results, current_it_gen_config)
        else:
            compare_generation_results_vs_ref(ov_prompts[i], ref[i], ov_results)


class GenerationChatInputsType(Enum):
    STRING = 1,
    ENCODED_INPUTS = 2,
    CHAT_HISTORY = 3
