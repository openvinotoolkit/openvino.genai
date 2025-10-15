# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datasets
import pytest
from dataclasses import dataclass
from tqdm import tqdm

from openvino_genai import ContinuousBatchingPipeline, GenerationConfig, CacheEvictionConfig, AggregationMode, KVCrushAnchorPointMode, KVCrushConfig

from utils.longbench import dataset2maxlen, evaluate, preprocess_prompt, post_process_pred
from utils.constants import get_default_llm_properties
from utils.hugging_face import download_and_convert_model
from kv_cache_eviction_utils import get_scheduler_config


LONGBENCH_CACHE_EVICTION_CONFIG = CacheEvictionConfig(start_size=32, recent_size=128, max_cache_size=672, aggregation_mode=AggregationMode.NORM_SUM)


@dataclass
class LongBenchTestData:
    subset: str
    threshold: float
    max_cache_usage_optimization_ratio: float
    avg_cache_usage_optimization_ratio: float


@pytest.mark.precommit
@pytest.mark.parametrize("test_struct", [
    LongBenchTestData("samsum", 4, 1.6, 2.5),
    LongBenchTestData("trec", 3.2, 2.0, 3.3),
], ids=["samsum", "trec"])
@pytest.mark.cache_eviction_part2
def test_optimized_generation_longbench(test_struct):
    seqs_per_request = 32
    device = "CPU"
    num_kv_blocks = 1000 if device == "CPU" else 500
    model_id = "Qwen/Qwen2-0.5B-Instruct"
    _, _, models_path = download_and_convert_model(model_id)
    scheduler_config = get_scheduler_config(num_kv_blocks)

    scheduler_config_opt = get_scheduler_config(num_kv_blocks)
    scheduler_config_opt.use_cache_eviction = True
    scheduler_config_opt.cache_eviction_config = LONGBENCH_CACHE_EVICTION_CONFIG

    scheduler_config_opt.use_sparse_attention = True

    model_cb_noopt = ContinuousBatchingPipeline(models_path, scheduler_config, device, {}, get_default_llm_properties())
    model_cb_opt = ContinuousBatchingPipeline(models_path, scheduler_config_opt, device, {}, get_default_llm_properties())

    model_name = "/".join(models_path.parts[-2:])
    subset = test_struct.subset
    max_new_tokens = dataset2maxlen[subset]

    generation_config = GenerationConfig()  # expecting default greedy sampling
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = max_new_tokens

    data = datasets.load_dataset('THUDM/LongBench', subset, split='test[:32]', trust_remote_code=True)
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

    del model_cb_opt
    del model_cb_noopt
    import gc
    gc.collect()

    assert ref_score - score <= test_struct.threshold
    assert max_optimization_ratio >= test_struct.max_cache_usage_optimization_ratio
    assert avg_optimization_ratio >= test_struct.avg_cache_usage_optimization_ratio


# KVCrush test configurations
KVCRUSH_SNAPKV_BASELINE_CONFIG = CacheEvictionConfig(
    start_size=32, 
    recent_size=128, 
    max_cache_size=1024, 
    aggregation_mode=AggregationMode.NORM_SUM,
    apply_rotation=False,
    snapkv_window_size=8,
    kvcrush_config=KVCrushConfig(budget=0)
)


OPTIMAL_KVCRUSH_CONFIGS = {
    "samsum": (768, 8, KVCrushAnchorPointMode.ALTERNATE),
    "trec": (960, 2, KVCrushAnchorPointMode.ALTERNATE), 
    "qasper": (960, 2, KVCrushAnchorPointMode.ALTERNATE)
}


@pytest.mark.precommit
@pytest.mark.parametrize("subset", ["samsum", "trec", "qasper"])
@pytest.mark.cache_eviction_part2
def test_kvcrush_vs_snapkv_baseline(subset):
    """Test that KVCrush performs equal or better than SnapKV baseline on LongBench datasets."""
    device = "CPU"
    seqs_per_request = 32
    num_kv_blocks = 1000 if device == "CPU" else 500
    model_id = "Qwen/Qwen2-0.5B-Instruct"
    _, _, models_path = download_and_convert_model(model_id)

    # Setup baseline and KVCrush configurations
    scheduler_config_baseline = get_scheduler_config(num_kv_blocks)
    scheduler_config_baseline.use_cache_eviction = True
    scheduler_config_baseline.cache_eviction_config = KVCRUSH_SNAPKV_BASELINE_CONFIG

    scheduler_config_kvcrush = get_scheduler_config(num_kv_blocks)
    scheduler_config_kvcrush.use_cache_eviction = True
    max_cache_size, budget, anchor_mode = OPTIMAL_KVCRUSH_CONFIGS[subset]
    config = CacheEvictionConfig(
        start_size=32,
        recent_size=128,
        max_cache_size=max_cache_size,
        aggregation_mode=AggregationMode.NORM_SUM,
        apply_rotation=False,
        snapkv_window_size=8,
        kvcrush_config=KVCrushConfig(budget=budget, anchor_point_mode=anchor_mode)
    )
    scheduler_config_kvcrush.cache_eviction_config = config

    model_cb_baseline = ContinuousBatchingPipeline(models_path, scheduler_config_baseline, device, {}, get_default_llm_properties())
    model_cb_kvcrush = ContinuousBatchingPipeline(models_path, scheduler_config_kvcrush, device, {}, get_default_llm_properties())

    model_name = "/".join(models_path.parts[-2:])
    max_new_tokens = dataset2maxlen[subset]

    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = max_new_tokens
    generation_config.apply_chat_template = False

    data = datasets.load_dataset('THUDM/LongBench', subset, split='test[:32]')
    with tqdm(total=len(data)) as progress_bar:
        batch = []
        baseline_answers = []
        kvcrush_answers = []
        for p_idx, data_sample in enumerate(data):
            prompt = preprocess_prompt(data_sample, subset, model_name)
            progress_bar.update(1)
            batch.append(prompt)
            baseline_answers.append({"answers": data_sample["answers"], "all_classes": data_sample["all_classes"]})
            kvcrush_answers.append({"answers": data_sample["answers"], "all_classes": data_sample["all_classes"]})

            if len(batch) == seqs_per_request or p_idx == len(data) - 1:
                baseline_batch = model_cb_baseline.generate(
                    batch, [generation_config] * len(batch)
                )
                kvcrush_batch = model_cb_kvcrush.generate(
                    batch, [generation_config] * len(batch)
                )
                for i, (baseline_output, kvcrush_output) in enumerate(zip(baseline_batch, kvcrush_batch), start=p_idx-len(batch)+1):
                    baseline_answers[i]["pred"] = post_process_pred(baseline_output.m_generation_ids[0], subset, model_name)
                    kvcrush_answers[i]["pred"] = post_process_pred(kvcrush_output.m_generation_ids[0], subset, model_name)
                batch.clear()

    baseline_score = evaluate(baseline_answers, subset)
    kvcrush_score = evaluate(kvcrush_answers, subset)

    print(f"Baseline (SnapKV) score: {baseline_score}")
    print(f"KVCrush score: {kvcrush_score}")

    assert kvcrush_score >= baseline_score, f"KVCrush score ({kvcrush_score}) is worse than baseline ({baseline_score}) on {subset} dataset"

    del model_cb_baseline
    del model_cb_kvcrush
    import gc
    gc.collect()
