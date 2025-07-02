# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import datasets
import pytest
import os
import csv
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from openvino_genai import ContinuousBatchingPipeline, SchedulerConfig, GenerationConfig, CacheEvictionConfig, AggregationMode, KVCrushAnchorPointMode, KVCrushConfig

from utils.ov_genai_pipelines import PipelineType, generate_and_compare
from utils.longbench import dataset2maxlen, evaluate, preprocess_prompt, post_process_pred
from utils.constants import get_default_llm_properties
from utils.hugging_face import download_and_convert_model
from data.test_dataset import get_test_dataset


def load_prompts_dataset(file_name : str) -> dict[str, list[str]]:
    TESTS_ROOT = Path(__file__).parent
    file_path = TESTS_ROOT / 'data' / file_name
    with open(file_path, 'r', encoding="utf-8") as f:
        return {"prompts": [s for s in f]}

def get_scheduler_config(num_kv_blocks: int) -> SchedulerConfig:
    scheduler_config = SchedulerConfig()
    scheduler_config.num_kv_blocks = num_kv_blocks
    scheduler_config.dynamic_split_fuse = True
    scheduler_config.max_num_batched_tokens = 256
    scheduler_config.max_num_seqs = 256
    scheduler_config.use_cache_eviction = False
    return scheduler_config


# Define multiple cache eviction configurations for testing
LONGBENCH_CACHE_EVICTION_CONFIGS = [
    # Config 1: Default RANDOM mode
    CacheEvictionConfig(
        start_size=32, 
        recent_size=128, 
        max_cache_size=512, 
        aggregation_mode=AggregationMode.NORM_SUM,
        apply_rotation=False,
        snapkv_window_size=1,
        kvcrush_config=KVCrushConfig(budget=0)
    ),
    # Config 2: KVCrush(max_cache_size=480, budget=1, anchor_point_mode=MEAN)
    CacheEvictionConfig(
        start_size=32, 
        recent_size=128, 
        max_cache_size=448, 
        aggregation_mode=AggregationMode.NORM_SUM,
        apply_rotation=False,
        snapkv_window_size=1,
        kvcrush_config=KVCrushConfig(budget=2, anchor_point_mode=KVCrushAnchorPointMode.MEAN)
    ),
    # Config 3: KVCrush(max_cache_size=480, budget=1, anchor_point_mode=ALTERNATE)
    CacheEvictionConfig(
        start_size=32, 
        recent_size=128, 
        max_cache_size=448, 
        aggregation_mode=AggregationMode.NORM_SUM,
        apply_rotation=False,
        snapkv_window_size=1,
        kvcrush_config=KVCrushConfig(budget=2, anchor_point_mode=KVCrushAnchorPointMode.ALTERNATE)
    ),
    # Config 4: KVCrush(max_cache_size=480, budget=1, anchor_point_mode=RANDOM)
    CacheEvictionConfig(
        start_size=32, 
        recent_size=128, 
        max_cache_size=448, 
        aggregation_mode=AggregationMode.NORM_SUM,
        apply_rotation=False,
        snapkv_window_size=1,
        kvcrush_config=KVCrushConfig(budget=2, anchor_point_mode=KVCrushAnchorPointMode.RANDOM)
    )
]

@dataclass
class LongBenchTestData:
    subset: str
    config_name: str
    config_idx: int


@pytest.mark.nightly
@pytest.mark.parametrize("device", ["CPU"])
@pytest.mark.parametrize("test_struct", [
    # Run each dataset with each configuration   

    *[LongBenchTestData(subset="qasper", 
                        config_name=f"Config{i+1}_{cfg.kvcrush_config.anchor_point_mode.name}_B{cfg.kvcrush_config.budget}", 
                        config_idx=i)
      for i, cfg in enumerate(LONGBENCH_CACHE_EVICTION_CONFIGS)],

    *[LongBenchTestData(subset="samsum", 
                        config_name=f"Config{i+1}_{cfg.kvcrush_config.anchor_point_mode.name}_B{cfg.kvcrush_config.budget}", 
                        config_idx=i)
      for i, cfg in enumerate(LONGBENCH_CACHE_EVICTION_CONFIGS)],
    
    *[LongBenchTestData(subset="trec", 
                        config_name=f"Config{i+1}_{cfg.kvcrush_config.anchor_point_mode.name}_B{cfg.kvcrush_config.budget}", 
                        config_idx=i)
      for i, cfg in enumerate(LONGBENCH_CACHE_EVICTION_CONFIGS)],
])
def test_optimized_generation_longbench(device, test_struct):
    seqs_per_request = 32
    num_kv_blocks = 1000 if device == "CPU" else 500
    model_id = "Qwen/Qwen2-0.5B-Instruct"
    _, _, models_path = download_and_convert_model(model_id)
    scheduler_config = get_scheduler_config(num_kv_blocks)

    scheduler_config_opt = get_scheduler_config(num_kv_blocks)
    scheduler_config_opt.use_cache_eviction = True
    if scheduler_config_opt.use_cache_eviction:
        # Use the specified configuration
        scheduler_config_opt.cache_eviction_config = LONGBENCH_CACHE_EVICTION_CONFIGS[test_struct.config_idx]
    
    print(f"\n\n========================================")
    print(f"Running test with {test_struct.config_name} on {test_struct.subset} dataset")
    print(f"KVCrush mode: {LONGBENCH_CACHE_EVICTION_CONFIGS[test_struct.config_idx].kvcrush_config.anchor_point_mode.name}")
    print(f"KVCrush budget: {LONGBENCH_CACHE_EVICTION_CONFIGS[test_struct.config_idx].kvcrush_config.budget}")
    print(f"========================================\n")

    model_cb_noopt = ContinuousBatchingPipeline(models_path, scheduler_config, device, {}, get_default_llm_properties())
    model_cb_opt = ContinuousBatchingPipeline(models_path, scheduler_config_opt, device, {}, get_default_llm_properties())

    model_name = "/".join(models_path.parts[-2:])
    subset = test_struct.subset
    max_new_tokens = dataset2maxlen[subset]

    generation_config = GenerationConfig()  # expecting default greedy sampling
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = max_new_tokens

    data = datasets.load_dataset('THUDM/LongBench', subset, split='test[:1000]')
    with tqdm(total=len(data)) as progress_bar:
        batch = []
        answers = []
        ref_answers = []
        for p_idx, data_sample in enumerate(data):
            prompt = preprocess_prompt(data_sample, subset, model_name)
            progress_bar.update(1)
            batch.append(prompt)
            answers.append({"answers": data_sample["answers"], "all_classes": data_sample["all_classes"]})
            ref_answers.append({"answers": data_sample["answers"], "all_classes": data_sample["all_classes"]})

            if len(batch) == seqs_per_request or p_idx == len(data) - 1:
                ans_batch = model_cb_opt.generate(
                    batch, [generation_config] * len(batch)
                )
                ref_ans_batch = model_cb_noopt.generate(
                    batch, [generation_config] * len(batch)
                )
                for i, (opt_output, ref_output) in enumerate(zip(ans_batch, ref_ans_batch), start=p_idx-len(batch)+1):
                    answers[i]["pred"] = post_process_pred(opt_output.m_generation_ids[0], subset, model_name)
                    ref_answers[i]["pred"] = post_process_pred(ref_output.m_generation_ids[0], subset, model_name)
                batch.clear()

    score = evaluate(answers, subset)
    print(f"Score: {score}")

    ref_score = evaluate(ref_answers, subset)
    print(f"Reference score: {ref_score}")
    pipeline_opt_metrics = model_cb_opt.get_metrics()
    pipeline_noopt_metrics = model_cb_noopt.get_metrics()

    print(f"No-opt cache usage: max {pipeline_noopt_metrics.max_cache_usage:.3f}, avg {pipeline_noopt_metrics.avg_cache_usage:.3f}")
    print(f"Opt cache usage: max {pipeline_opt_metrics.max_cache_usage:.3f}, avg {pipeline_opt_metrics.avg_cache_usage:.3f}")
    max_optimization_ratio = (pipeline_noopt_metrics.max_cache_usage / pipeline_opt_metrics.max_cache_usage)
    avg_optimization_ratio = (pipeline_noopt_metrics.avg_cache_usage / pipeline_opt_metrics.avg_cache_usage)
    print(f"Optimization ratios: max {max_optimization_ratio:.3f}x, avg {avg_optimization_ratio:.3f}x")
    
    # Save results to CSV
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"results/kvcrush_results.csv"
    file_exists = os.path.isfile(result_file)
    
    with open(result_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Subset', 'Config', 'Mode', 'Budget', 'Score', 'RefScore', 
                            'MaxCacheUsage', 'AvgCacheUsage', 'MaxOptRatio', 'AvgOptRatio', 'ScoreDiff'])
        
        config = LONGBENCH_CACHE_EVICTION_CONFIGS[test_struct.config_idx]
        writer.writerow([
            timestamp,
            subset,
            test_struct.config_name,
            config.kvcrush_config.anchor_point_mode.name,
            config.kvcrush_config.budget,
            score,
            ref_score,
            pipeline_opt_metrics.max_cache_usage,
            pipeline_opt_metrics.avg_cache_usage,
            max_optimization_ratio,
            avg_optimization_ratio,
            ref_score - score
        ])
    
    print(f"Results saved to {result_file}")

    del model_cb_opt
    del model_cb_noopt
    import gc
    gc.collect()