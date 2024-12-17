# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional

import pytest

from optimum.intel.openvino import OVModelForCausalLM

from openvino_genai import ContinuousBatchingPipeline, SchedulerConfig, GenerationResult, GenerationConfig, CacheEvictionConfig, AggregationMode

from openvino_tokenizers import convert_tokenizer
from openvino import serialize
from transformers import AutoTokenizer

from common import TESTS_ROOT


def load_prompts_dataset(file_name : str) -> Dict[str, List[str]]:
    file_path = TESTS_ROOT / 'data' / file_name
    with open(file_path, 'r') as f:
        return {"prompts": [s for s in f]}

def get_scheduler_config(num_kv_blocks: int) -> SchedulerConfig:
    scheduler_config = SchedulerConfig()
    scheduler_config.num_kv_blocks = num_kv_blocks
    scheduler_config.dynamic_split_fuse = True
    scheduler_config.max_num_batched_tokens = 256
    scheduler_config.max_num_seqs = 256
    scheduler_config.use_cache_eviction = False
    return scheduler_config

@dataclass
class ConvertedModel:
    model: OVModelForCausalLM
    tokenizer: AutoTokenizer
    models_path: Path


@pytest.fixture(scope='module')
def converted_model(tmp_path_factory):
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    models_path = tmp_path_factory.mktemp("cacheopt_test_models") / model_id
    model.save_pretrained(models_path)
    ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True, skip_special_tokens=True)
    serialize(ov_tokenizer, models_path / "openvino_tokenizer.xml")
    serialize(ov_detokenizer, models_path / "openvino_detokenizer.xml")
    converted_model = ConvertedModel(model, tokenizer, models_path)
    yield converted_model
    del converted_model
    del model


@dataclass
class CacheOptTestStruct:
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
@pytest.mark.skipif(sys.platform in ("win32", "darwin"), reason="doesn't work on win due to optimum-intel export bug, segfault on mac")
@pytest.mark.parametrize("test_struct", [
    # prompts + generation length are longer than the eviction arena, eviction expected w/ impact to similarity
    CacheOptTestStruct(prompt_file="long_prompts.txt", max_new_tokens=128, num_kv_blocks=1000, use_cache_eviction=True,
                       cache_eviction_config=SHORT_CACHE_EVICTION_CONFIG,
                       similarity_threshold=0.8,
                       max_cache_usage_optimization_ratio=2.0,
                       avg_cache_usage_optimization_ratio=1.7),

    # prompts + generation length are shorter than the eviction arena, no eviction expected
    CacheOptTestStruct(prompt_file="short_prompts.txt", max_new_tokens=32, num_kv_blocks=1000, use_cache_eviction=True,
                       cache_eviction_config=SHORT_CACHE_EVICTION_CONFIG,
                       similarity_threshold=0.98,
                       max_cache_usage_optimization_ratio=0.95,  # no improvement expected
                       avg_cache_usage_optimization_ratio=0.95),

    # short prompts, long generation - eviction expected
    CacheOptTestStruct(prompt_file="short_prompts.txt", max_new_tokens=160, num_kv_blocks=1000, use_cache_eviction=True,
                       cache_eviction_config=SHORT_CACHE_EVICTION_CONFIG,
                       similarity_threshold=0.94,
                       max_cache_usage_optimization_ratio=1.4,
                       avg_cache_usage_optimization_ratio=1.1),

])
@pytest.mark.parametrize("enable_prefix_caching", [True, False])  # prefix caching shouldn't impact similarity
def test_cache_optimized_generation_is_similar_to_unoptimized(converted_model, test_struct, enable_prefix_caching):
    import whowhatbench
    
    seqs_per_request = 32
    scheduler_config = get_scheduler_config(test_struct.num_kv_blocks)

    generation_config = GenerationConfig()  # expecting default greedy sampling
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = test_struct.max_new_tokens

    scheduler_config_opt = get_scheduler_config(test_struct.num_kv_blocks)
    scheduler_config_opt.use_cache_eviction = test_struct.use_cache_eviction
    if scheduler_config_opt.use_cache_eviction:
        scheduler_config_opt.cache_eviction_config = test_struct.cache_eviction_config
    scheduler_config_opt.enable_prefix_caching = enable_prefix_caching

    models_path = converted_model.models_path
    model_cb_noopt = ContinuousBatchingPipeline(models_path.absolute().as_posix(), scheduler_config, "CPU", {})
    model_cb_opt = ContinuousBatchingPipeline(models_path.absolute().as_posix(), scheduler_config_opt, "CPU", {})

    tokenizer = converted_model.tokenizer

    data_dict = load_prompts_dataset(test_struct.prompt_file)

    evaluator = whowhatbench.TextEvaluator(base_model=model_cb_noopt, tokenizer=tokenizer, test_data=data_dict,
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

    assert similarity_metric > test_struct.similarity_threshold
    assert max_optimization_ratio >= test_struct.max_cache_usage_optimization_ratio
    assert avg_optimization_ratio >= test_struct.avg_cache_usage_optimization_ratio

    del model_cb_opt
    del model_cb_noopt


