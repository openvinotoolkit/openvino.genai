# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datasets
import pytest

from tqdm import tqdm

from openvino_genai import ContinuousBatchingPipeline, GenerationConfig, CacheEvictionConfig, AggregationMode, KVCrushAnchorPointMode, KVCrushConfig

from utils.longbench import dataset2maxlen, evaluate, preprocess_prompt, post_process_pred
from utils.constants import get_default_llm_properties
from utils.hugging_face import download_and_convert_model
from kv_cache_eviction_utils import get_scheduler_config


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
def test_kvcrush_vs_snapkv_baseline_longbench(subset):
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
