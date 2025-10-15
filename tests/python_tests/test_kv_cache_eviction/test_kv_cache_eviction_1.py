# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openvino_genai import ContinuousBatchingPipeline, GenerationConfig, CacheEvictionConfig, AggregationMode

from utils.ov_genai_pipelines import PipelineType, generate_and_compare
from utils.constants import get_default_llm_properties
from utils.hugging_face import download_and_convert_model
from data.test_dataset import get_test_dataset
from kv_cache_eviction_utils import get_scheduler_config


def load_prompts_dataset(file_name : str) -> dict[str, list[str]]:
    TESTS_ROOT = Path(__file__).parent
    file_path = TESTS_ROOT / 'data' / file_name
    with open(file_path, 'r', encoding="utf-8") as f:
        return {"prompts": [s for s in f]}


@dataclass
class CacheOptTestStruct:
    test_id: str
    prompt_file: str
    max_new_tokens: int
    num_kv_blocks: int
    use_cache_eviction: bool
    cache_eviction_config: Optional[CacheEvictionConfig]
    similarity_threshold: float
    avg_cache_usage_optimization_ratio: float  # expecting no less than these optimization ratios
    max_cache_usage_optimization_ratio: float


SHORT_CACHE_EVICTION_CONFIG = CacheEvictionConfig(start_size=32, recent_size=32, max_cache_size=96, aggregation_mode=AggregationMode.NORM_SUM)

@pytest.mark.precommit
@pytest.mark.skipif(
    sys.platform in ("win32", "darwin"),
    reason=(
        "doesn't work on win due to optimum-intel export bug, "
        "segfault on mac"
    ),
)
@pytest.mark.parametrize("test_struct", [
    # prompts + generation length are longer than the eviction arena, eviction expected w/ impact to similarity
    CacheOptTestStruct(test_id="prompts_longer_than_eviction_arena",
                       prompt_file="long_prompts.txt", max_new_tokens=128, num_kv_blocks=500, use_cache_eviction=True,
                       cache_eviction_config=SHORT_CACHE_EVICTION_CONFIG,
                       similarity_threshold=0.8,
                       max_cache_usage_optimization_ratio=2.0,
                       avg_cache_usage_optimization_ratio=1.7),

    # prompts + generation length are shorter than the eviction arena, no eviction expected
    CacheOptTestStruct(test_id="prompts_and_gen_shorter_than_eviction_arena",
                       prompt_file="short_prompts.txt", max_new_tokens=32, num_kv_blocks=500, use_cache_eviction=True,
                       cache_eviction_config=SHORT_CACHE_EVICTION_CONFIG,
                       similarity_threshold=0.98,
                       max_cache_usage_optimization_ratio=0.95,  # no improvement expected
                       avg_cache_usage_optimization_ratio=0.95),

    # short prompts, long generation - eviction expected
    CacheOptTestStruct(test_id="gen_longer_than_eviction_arena",
                       prompt_file="short_prompts.txt", max_new_tokens=160, num_kv_blocks=500, use_cache_eviction=True,
                       cache_eviction_config=SHORT_CACHE_EVICTION_CONFIG,
                       similarity_threshold=0.94,
                       max_cache_usage_optimization_ratio=1.4,
                       avg_cache_usage_optimization_ratio=1.1),

    ], ids=lambda x: x.test_id)
@pytest.mark.parametrize("apply_rotation", [True, False], ids=["with_rotation", "no_rotation"])         # rotation should improve similarity
@pytest.mark.parametrize("use_sparse_attention", [True, False], ids=["with_sparse_attn", "no_sparse_attn"]) # sparse attn should not degrade similarity too much
def test_cache_optimized_generation_is_similar_to_unoptimized(test_struct, apply_rotation, use_sparse_attention):
    import whowhatbench

    seqs_per_request = 32
    scheduler_config = get_scheduler_config(test_struct.num_kv_blocks)

    generation_config = GenerationConfig()  # expecting default greedy sampling
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = test_struct.max_new_tokens
    generation_config.apply_chat_template = False

    scheduler_config_opt = get_scheduler_config(test_struct.num_kv_blocks)
    scheduler_config_opt.use_cache_eviction = test_struct.use_cache_eviction
    if scheduler_config_opt.use_cache_eviction:
        scheduler_config_opt.cache_eviction_config = test_struct.cache_eviction_config
        scheduler_config_opt.cache_eviction_config.apply_rotation = apply_rotation
    scheduler_config_opt.use_sparse_attention = use_sparse_attention
    if use_sparse_attention:
        scheduler_config_opt.sparse_attention_config.num_last_dense_tokens_in_prefill = 10

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    _, tokenizer, models_path = download_and_convert_model(model_id)
    model_cb_noopt = ContinuousBatchingPipeline(models_path, scheduler_config, "CPU", {}, get_default_llm_properties())
    model_cb_opt = ContinuousBatchingPipeline(models_path, scheduler_config_opt, "CPU", {}, get_default_llm_properties())

    data_dict = load_prompts_dataset(test_struct.prompt_file)

    evaluator = whowhatbench.Evaluator(base_model=model_cb_noopt, tokenizer=tokenizer, test_data=data_dict,
                                       generation_config=generation_config,
                                       generation_config_base=generation_config,
                                       max_new_tokens=test_struct.max_new_tokens, seqs_per_request=seqs_per_request)

    _, all_metrics = evaluator.score(model_cb_opt)

    similarity_metric = float(all_metrics['similarity'][0])
    pipeline_opt_metrics = model_cb_opt.get_metrics()
    pipeline_noopt_metrics = model_cb_noopt.get_metrics()

    print(f"Similarity: {similarity_metric}")
    print(f"No-opt cache usage: max {pipeline_noopt_metrics.max_cache_usage:.3f}, avg {pipeline_noopt_metrics.avg_cache_usage:.3f}")
    print(f"Opt cache usage: max {pipeline_opt_metrics.max_cache_usage:.3f}, avg {pipeline_opt_metrics.avg_cache_usage:.3f}")
    max_optimization_ratio = (pipeline_noopt_metrics.max_cache_usage / pipeline_opt_metrics.max_cache_usage)
    avg_optimization_ratio = (pipeline_noopt_metrics.avg_cache_usage / pipeline_opt_metrics.avg_cache_usage)
    print(f"Optimization ratios: max {max_optimization_ratio:.3f}x, avg {avg_optimization_ratio:.3f}x")

    del model_cb_opt
    del model_cb_noopt
    del evaluator
    del data_dict
    import gc
    gc.collect()  # without gc.collect() each case leaks 300 MB of RAM

    is_similar = similarity_metric > test_struct.similarity_threshold

    if apply_rotation and not is_similar:
        pytest.xfail("cache rotation currently has worse similarity due to unknown reasons")

    assert similarity_metric > test_struct.similarity_threshold
    assert max_optimization_ratio >= test_struct.max_cache_usage_optimization_ratio
    assert avg_optimization_ratio >= test_struct.avg_cache_usage_optimization_ratio



def get_greedy_seq_len_300() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.max_new_tokens = 300
    return generation_config


def get_beam_search_seq_len_300() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_beam_groups = 3
    generation_config.num_beams = 6
    generation_config.diversity_penalty = 1
    generation_config.max_new_tokens = 300
    generation_config.num_return_sequences = generation_config.num_beams
    return generation_config


scheduler_params_list = [
                         ({"num_kv_blocks": 0, "cache_size": 0, "dynamic_split_fuse": True, "enable_prefix_caching": True}, get_greedy_seq_len_300()),
                         ({"num_kv_blocks": 0, "cache_size": 0, "dynamic_split_fuse": False, "max_num_batched_tokens": 600, "enable_prefix_caching": True}, get_beam_search_seq_len_300()),
                         ({"num_kv_blocks": 0, "cache_size": 0, "dynamic_split_fuse": True, "enable_prefix_caching": False}, get_greedy_seq_len_300()),
                         ({"num_kv_blocks": 0, "cache_size": 0, "dynamic_split_fuse": False, "max_num_batched_tokens": 600, "enable_prefix_caching": False}, get_beam_search_seq_len_300()),
                         ({"num_kv_blocks": 0, "cache_size": 0, "dynamic_split_fuse": False, "max_num_batched_tokens": 600, "use_cache_eviction": True, "cache_eviction_config": SHORT_CACHE_EVICTION_CONFIG}, get_greedy_seq_len_300())]
@pytest.mark.parametrize("params", scheduler_params_list)
@pytest.mark.precommit
def test_dynamic_memory_allocation(params):
    prompts, _ = get_test_dataset()
    generate_and_compare(prompts=prompts,
                         model="facebook/opt-125m",
                         scheduler_config=params[0],
                         generation_config=params[1],
                         pipeline_type=PipelineType.CONTINUOUS_BATCHING)
