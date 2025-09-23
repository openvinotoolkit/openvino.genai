# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import datasets
import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from optimum.intel.openvino import OVModelForVisualCausalLM
from openvino_genai import ContinuousBatchingPipeline, SchedulerConfig, GenerationConfig, CacheEvictionConfig, AggregationMode, SparseAttentionMode, KVCrushAnchorPointMode, KVCrushConfig

from utils.ov_genai_pipelines import PipelineType, generate_and_compare
from utils.longbench import dataset2maxlen, evaluate, preprocess_prompt, post_process_pred
from utils.milebench import MileBenchDataset, Eval
from utils.constants import get_default_llm_properties
from utils.hugging_face import download_and_convert_model, _download_and_convert_model
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
LONGBENCH_CACHE_EVICTION_CONFIG = CacheEvictionConfig(start_size=32, recent_size=128, max_cache_size=672, aggregation_mode=AggregationMode.NORM_SUM)

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


@dataclass
class BenchmarkTestData:
    subset: str
    threshold: float
    max_cache_usage_optimization_ratio: float
    avg_cache_usage_optimization_ratio: float


@pytest.mark.precommit
@pytest.mark.parametrize("test_struct", [
    BenchmarkTestData("samsum", 4, 1.6, 2.5),
    BenchmarkTestData("trec", 3.2, 2.0, 3.3),
], ids=["samsum", "trec"])
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


@pytest.mark.precommit
@pytest.mark.parametrize("subset", ["samsum", "trec", "qasper"])
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


MILEBENCH_CACHE_EVICTION_CONFIG = CacheEvictionConfig(start_size=32, recent_size=64, max_cache_size=352, aggregation_mode=AggregationMode.SUM)


@pytest.mark.precommit
@pytest.mark.parametrize(
    ("test_struct", "download_test_content"), [
    (BenchmarkTestData("ALFRED", 0.008, 1.98, 1.91), "MileBench_part0.tar.gz"),
    (BenchmarkTestData("MMCoQA", 0.001, 3.75, 4.55), "MileBench_part2.tar.gz"),
    ],
    indirect=["download_test_content"],
    ids=["ALFRED", "MMCoQA"],
)
def test_optimized_generation_milebench(test_struct, download_test_content):
    seqs_per_request = 16
    device = "CPU"
    num_kv_blocks = 500
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    _, _, models_path = _download_and_convert_model(model_id, OVModelForVisualCausalLM)
    scheduler_config = get_scheduler_config(num_kv_blocks)

    scheduler_config_opt = get_scheduler_config(num_kv_blocks)
    scheduler_config_opt.use_cache_eviction = True
    if scheduler_config_opt.use_cache_eviction:
        eviction_config = CacheEvictionConfig(
            start_size=32,
            recent_size=64,
            max_cache_size=224,
            aggregation_mode=AggregationMode.SUM,
            snapkv_window_size=8,
        )
        scheduler_config_opt.cache_eviction_config = eviction_config

    model_cb_noopt = ContinuousBatchingPipeline(models_path, scheduler_config, device, properties=get_default_llm_properties())
    model_cb_opt = ContinuousBatchingPipeline(models_path, scheduler_config_opt, device, properties=get_default_llm_properties())

    generation_config = GenerationConfig()  # expecting default greedy sampling
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 64  # change to 512 for full evaluation
    generation_config.do_sample = False
    generation_config.apply_chat_template = False

    subset = test_struct.subset
    data = MileBenchDataset(
        data_dir=download_test_content,
        subset=subset,
        subset_size=seqs_per_request,
    )

    with tqdm(total=len(data)) as progress_bar:
        prompts, images = [], []
        answers = []
        ref_answers = []
        for p_idx, data_sample in enumerate(data):
            prompt = data_sample["prompt"]
            image = data_sample["images"]

            progress_bar.update(1)
            prompts.append(prompt)
            images.append(image)
            answers.append({"gt_answer": data_sample["gt_answer"], "choice_list": data_sample["choice_list"]})
            ref_answers.append({"gt_answer": data_sample["gt_answer"], "choice_list": data_sample["choice_list"]})

            if len(prompts) == seqs_per_request or p_idx == len(data) - 1:
                ans_batch = model_cb_opt.generate(
                    prompts, images=images, generation_config=[generation_config] * len(prompts),
                )
                ref_ans_batch = model_cb_noopt.generate(
                    prompts, images=images, generation_config=[generation_config] * len(prompts),
                )

                for i, (opt_output, ref_output) in enumerate(zip(ans_batch, ref_ans_batch), start=p_idx-len(prompts)+1):
                    answers[i]["pred"] = opt_output.texts[0]
                    ref_answers[i]["pred"] = ref_output.texts[0]
                prompts.clear()
                images.clear()

    question_type = data.annotation['meta_data']['question_type']
    scorer = Eval()

    score = scorer.evaluate(answers, subset, question_type)
    print(f"Score: {score}")

    ref_score = scorer.evaluate(ref_answers, subset, question_type)
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
