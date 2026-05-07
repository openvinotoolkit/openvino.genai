# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import gc
from pathlib import Path

import pytest


def _local_dflash_models():
    if os.environ.get("OV_GENAI_ENABLE_DFLASH_E2E") != "1":
        pytest.skip("Set OV_GENAI_ENABLE_DFLASH_E2E=1 to run local DFlash artifact tests")

    repo_root = Path(__file__).resolve().parents[3]
    target_model_path = repo_root / "models" / "qwen3-coder-30b-a3b-instruct-int4-annotated-ov"
    draft_model_path = repo_root / "models" / "qwen3-coder-30b-a3b-dflash-ov"
    if not target_model_path.exists() or not draft_model_path.exists():
        pytest.skip("Local DFlash target/draft artifacts are not available")

    return target_model_path, draft_model_path


def _require_gpu_device():
    ov = pytest.importorskip("openvino")
    devices = ov.Core().get_available_devices()
    if not any(device == "GPU" or device.startswith("GPU.") for device in devices):
        pytest.skip(f"OpenVINO GPU device is not available, found devices: {devices}")


def _generate_target_then_dflash(device, prompts, generation_config):
    if device == "GPU":
        _require_gpu_device()
    target_model_path, draft_model_path = _local_dflash_models()
    ov_genai = pytest.importorskip("openvino_genai")

    target_pipe = ov_genai.LLMPipeline(target_model_path, device, {})
    target_results = [target_pipe.generate([prompt], generation_config) for prompt in prompts]
    target_snapshots = [
        (result.texts, result.perf_metrics.get_num_generated_tokens())
        for result in target_results
    ]

    # Release the baseline target before compiling the target+draft pair.
    del target_pipe
    del target_results
    gc.collect()

    draft = ov_genai.draft_model(draft_model_path, device)
    dflash_pipe = ov_genai.LLMPipeline(target_model_path, device, {}, draft_model=draft)
    dflash_results = [dflash_pipe.generate([prompt], generation_config) for prompt in prompts]
    del dflash_pipe
    del draft
    gc.collect()

    return target_snapshots, dflash_results


def _generate_target_then_dflash_on_gpu(prompts, generation_config):
    return _generate_target_then_dflash("GPU", prompts, generation_config)


def test_dflash_local_greedy_matches_target_pipeline():
    ov_genai = pytest.importorskip("openvino_genai")
    prompt = "Write a Python function that adds two integers."
    generation_config = ov_genai.GenerationConfig(do_sample=False, max_new_tokens=8, ignore_eos=True)

    target_snapshots, dflash_results = _generate_target_then_dflash("CPU", [prompt], generation_config)
    target_texts, target_num_generated = target_snapshots[0]
    dflash_result = dflash_results[0]

    assert dflash_result.texts == target_texts
    assert dflash_result.perf_metrics.get_num_generated_tokens() == target_num_generated


def test_dflash_local_perf_metrics_are_populated():
    target_model_path, draft_model_path = _local_dflash_models()
    ov_genai = pytest.importorskip("openvino_genai")
    draft = ov_genai.draft_model(draft_model_path, "CPU")
    pipe = ov_genai.LLMPipeline(target_model_path, "CPU", {}, draft_model=draft)

    result = pipe.generate(
        ["Explain speculative decoding briefly."],
        ov_genai.GenerationConfig(do_sample=False, max_new_tokens=8, ignore_eos=True),
    )
    assert result.extended_perf_metrics is not None
    assert result.extended_perf_metrics.main_model_metrics.raw_metrics.m_durations
    assert result.extended_perf_metrics.draft_model_metrics.raw_metrics.m_durations
    del pipe
    del draft
    gc.collect()


def test_dflash_local_gpu_greedy_matches_target_pipeline():
    ov_genai = pytest.importorskip("openvino_genai")
    generation_config = ov_genai.GenerationConfig(do_sample=False, max_new_tokens=8, ignore_eos=True)
    target_results, dflash_results = _generate_target_then_dflash_on_gpu(
        ["Write a Python function that adds two integers."],
        generation_config,
    )

    target_texts, target_num_generated = target_results[0]
    dflash_result = dflash_results[0]
    assert dflash_result.texts == target_texts
    assert dflash_result.perf_metrics.get_num_generated_tokens() == target_num_generated
    assert dflash_result.extended_perf_metrics is not None
    assert dflash_result.extended_perf_metrics.main_model_metrics.raw_metrics.m_durations
    assert dflash_result.extended_perf_metrics.draft_model_metrics.raw_metrics.m_durations


def test_dflash_local_gpu_exercises_target_kv_rollback():
    ov_genai = pytest.importorskip("openvino_genai")
    generation_config = ov_genai.GenerationConfig(do_sample=False, max_new_tokens=32, ignore_eos=True)
    prompts = [
        "Write a compact C++ function that checks if a number is prime.",
        "Explain speculative decoding in one paragraph.",
        "Continue this sequence with code-like tokens: def foo(x):",
    ]

    target_results, dflash_results = _generate_target_then_dflash_on_gpu(prompts, generation_config)

    total_draft_generated = 0
    total_draft_accepted = 0
    for prompt, target_snapshot, dflash_result in zip(prompts, target_results, dflash_results):
        target_texts, _ = target_snapshot
        assert dflash_result.texts == target_texts, f"DFlash diverged from target-only GPU for prompt: {prompt}"
        metrics = dflash_result.extended_perf_metrics
        assert metrics is not None
        draft_generated = metrics.draft_model_metrics.get_num_generated_tokens()
        draft_accepted = metrics.get_num_accepted_tokens()
        assert draft_generated > 0, f"DFlash draft did not generate candidates for prompt: {prompt}"
        assert draft_accepted <= draft_generated
        total_draft_generated += draft_generated
        total_draft_accepted += draft_accepted

    assert total_draft_generated > total_draft_accepted, (
        "GPU/GPU DFlash did not reject any draft candidates, so target KV-cache rollback was not exercised. "
        f"draft_generated={total_draft_generated}, draft_accepted={total_draft_accepted}"
    )
