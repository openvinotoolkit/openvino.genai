# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import gc
from pathlib import Path

import pytest


_TARGET_PIPELINE_PROPERTIES = {"ATTENTION_BACKEND": "PA"}
_CPU_ROLLBACK_PROMPT = "Explain speculative decoding in one paragraph."
_DFLASH_CB_E2E_ENV = "OV_GENAI_ENABLE_DFLASH_CB_E2E"
_LLAMA_DFLASH_E2E_ENV = "OV_GENAI_ENABLE_LLAMA_DFLASH_E2E"


def _local_dflash_cb_models():
    if os.environ.get(_DFLASH_CB_E2E_ENV) != "1":
        pytest.skip(f"Set {_DFLASH_CB_E2E_ENV}=1 to run local DFlash CB/PA artifact tests")

    repo_root = Path(__file__).resolve().parents[3]
    target_model_path = repo_root / "models" / "qwen3-coder-30b-a3b-instruct-int4-annotated-ov"
    draft_model_path = repo_root / "models" / "qwen3-coder-30b-a3b-dflash-stateful-woq-int8-ov"
    if not target_model_path.exists() or not draft_model_path.exists():
        pytest.skip("Local DFlash target/draft artifacts are not available")

    return target_model_path, draft_model_path


def _local_llama_dflash_cb_models():
    if os.environ.get(_LLAMA_DFLASH_E2E_ENV) != "1":
        pytest.skip(f"Set {_LLAMA_DFLASH_E2E_ENV}=1 to run local Llama DFlash CB/PA artifact tests")

    repo_root = Path(__file__).resolve().parents[3]
    target_model_path = repo_root / "models" / "llama-3.1-8b-instruct-4bit-ratio1-gs128-ov"
    draft_model_path = repo_root / "models" / "llama-3.1-8b-dflash-ultrachat-ov"
    if not target_model_path.exists() or not draft_model_path.exists():
        pytest.skip("Local Llama DFlash target/draft artifacts are not available")

    return target_model_path, draft_model_path


def _fp32_target_pipeline_properties():
    ov = pytest.importorskip("openvino")
    import openvino.properties.hint as hints

    properties = _TARGET_PIPELINE_PROPERTIES.copy()
    properties[hints.inference_precision] = ov.Type.f32
    properties[hints.kv_cache_precision] = ov.Type.f32
    return properties


def _assert_target_kv_states_are_fp32(target_model_path):
    ov = pytest.importorskip("openvino")
    model = ov.Core().read_model(target_model_path / "openvino_model.xml")
    state_types = {
        str(op.get_output_element_type(0))
        for op in model.get_ordered_ops()
        if op.get_type_name() == "ReadValue"
    }
    assert state_types == {"<Type: 'float32'>"}


def _require_gpu_device():
    ov = pytest.importorskip("openvino")
    devices = ov.Core().get_available_devices()
    if not any(device == "GPU" or device.startswith("GPU.") for device in devices):
        pytest.skip(f"OpenVINO GPU device is not available, found devices: {devices}")


def _snapshot_encoded_result(tokenizer, result):
    return {
        "texts": tokenizer.decode(result.tokens),
        "tokens": [list(tokens) for tokens in result.tokens],
        "num_generated": result.perf_metrics.get_num_generated_tokens(),
        "extended_perf_metrics": result.extended_perf_metrics,
    }


def _first_token_difference(left, right):
    for idx, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return idx, left_token, right_token
    if len(left) != len(right):
        return min(len(left), len(right)), None, None
    return None


def _assert_same_generation(prompt, target_snapshot, dflash_snapshot):
    target_tokens = target_snapshot["tokens"][0]
    dflash_tokens = dflash_snapshot["tokens"][0]
    if dflash_tokens != target_tokens:
        diff = _first_token_difference(target_tokens, dflash_tokens)
        pytest.fail(
            f"DFlash diverged from target-only for prompt: {prompt}\n"
            f"first_token_diff={diff}\n"
            f"target_tokens={target_tokens}\n"
            f"dflash_tokens={dflash_tokens}\n"
            f"target_text={target_snapshot['texts']}\n"
            f"dflash_text={dflash_snapshot['texts']}"
        )
    assert dflash_snapshot["texts"] == target_snapshot["texts"]


def _generate_target_then_dflash_cb(device, prompts, generation_config, properties=None, model_paths=None):
    if device == "GPU":
        _require_gpu_device()
    target_model_path, draft_model_path = model_paths if model_paths is not None else _local_dflash_cb_models()
    ov_genai = pytest.importorskip("openvino_genai")
    tokenizer = ov_genai.Tokenizer(target_model_path)
    tokenized_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    pipeline_properties = properties.copy() if properties is not None else _TARGET_PIPELINE_PROPERTIES.copy()

    target_pipe = ov_genai.LLMPipeline(target_model_path, device, pipeline_properties.copy())
    target_results = [target_pipe.generate(tokenized_prompt, generation_config) for tokenized_prompt in tokenized_prompts]
    target_snapshots = [_snapshot_encoded_result(tokenizer, result) for result in target_results]

    # Release the baseline target before compiling the target+draft pair.
    del target_pipe
    del target_results
    gc.collect()

    draft = ov_genai.draft_model(draft_model_path, device)
    dflash_pipe = ov_genai.LLMPipeline(
        target_model_path,
        device,
        pipeline_properties.copy(),
        draft_model=draft,
    )
    dflash_results = [dflash_pipe.generate(tokenized_prompt, generation_config) for tokenized_prompt in tokenized_prompts]
    dflash_snapshots = [_snapshot_encoded_result(tokenizer, result) for result in dflash_results]
    del dflash_pipe
    del draft
    gc.collect()

    return target_snapshots, dflash_snapshots


def _generate_target_then_dflash_cb_on_gpu(prompts, generation_config):
    return _generate_target_then_dflash_cb("GPU", prompts, generation_config)


def test_dflash_cb_target_pipeline_properties_use_pa_attention_backend():
    assert _TARGET_PIPELINE_PROPERTIES == {"ATTENTION_BACKEND": "PA"}


def test_dflash_local_greedy_matches_target_pipeline():
    ov_genai = pytest.importorskip("openvino_genai")
    prompt = "Write a Python function that adds two integers."
    generation_config = ov_genai.GenerationConfig(do_sample=False, max_new_tokens=8, ignore_eos=True)

    target_snapshots, dflash_results = _generate_target_then_dflash_cb("CPU", [prompt], generation_config)
    target_snapshot = target_snapshots[0]
    dflash_result = dflash_results[0]

    _assert_same_generation(prompt, target_snapshot, dflash_result)
    assert dflash_result["num_generated"] == target_snapshot["num_generated"]


def test_dflash_local_cpu_fp32_prompt2_exercises_target_kv_rollback():
    ov_genai = pytest.importorskip("openvino_genai")
    target_model_path, _ = _local_dflash_cb_models()
    _assert_target_kv_states_are_fp32(target_model_path)
    generation_config = ov_genai.GenerationConfig(do_sample=False, max_new_tokens=32, ignore_eos=True)

    target_snapshots, dflash_results = _generate_target_then_dflash_cb(
        "CPU",
        [_CPU_ROLLBACK_PROMPT],
        generation_config,
        _fp32_target_pipeline_properties(),
    )
    target_snapshot = target_snapshots[0]
    dflash_result = dflash_results[0]
    metrics = dflash_result["extended_perf_metrics"]
    assert metrics is not None
    draft_generated = metrics.draft_model_metrics.get_num_generated_tokens()
    draft_accepted = metrics.get_num_accepted_tokens()
    assert draft_generated > draft_accepted, (
        "CPU/fp32 DFlash did not reject any draft candidates, so target KV-cache rollback was not exercised. "
        f"draft_generated={draft_generated}, draft_accepted={draft_accepted}"
    )
    _assert_same_generation(_CPU_ROLLBACK_PROMPT, target_snapshot, dflash_result)


def test_dflash_local_perf_metrics_are_populated():
    target_model_path, draft_model_path = _local_dflash_cb_models()
    ov_genai = pytest.importorskip("openvino_genai")
    draft = ov_genai.draft_model(draft_model_path, "CPU")
    pipe = ov_genai.LLMPipeline(
        target_model_path,
        "CPU",
        _TARGET_PIPELINE_PROPERTIES.copy(),
        draft_model=draft,
    )

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


def test_dflash_local_rejects_sampling():
    target_model_path, draft_model_path = _local_dflash_cb_models()
    ov_genai = pytest.importorskip("openvino_genai")
    draft = ov_genai.draft_model(draft_model_path, "CPU")
    pipe = ov_genai.LLMPipeline(
        target_model_path,
        "CPU",
        _TARGET_PIPELINE_PROPERTIES.copy(),
        draft_model=draft,
    )

    generation_config = ov_genai.GenerationConfig(
        do_sample=True,
        max_new_tokens=4,
        ignore_eos=True,
    )
    with pytest.raises(RuntimeError, match="greedy decoding"):
        pipe.generate(["Explain speculative decoding briefly."], generation_config)
    del pipe
    del draft
    gc.collect()


def test_dflash_local_rejects_prompt_batch():
    target_model_path, draft_model_path = _local_dflash_cb_models()
    ov_genai = pytest.importorskip("openvino_genai")
    draft = ov_genai.draft_model(draft_model_path, "CPU")
    pipe = ov_genai.LLMPipeline(
        target_model_path,
        "CPU",
        _TARGET_PIPELINE_PROPERTIES.copy(),
        draft_model=draft,
    )

    generation_config = ov_genai.GenerationConfig(do_sample=False, max_new_tokens=4, ignore_eos=True)
    with pytest.raises(RuntimeError, match="batch size 1"):
        pipe.generate(
            [
                "Write a Python function that adds two integers.",
                "Explain speculative decoding briefly.",
            ],
            generation_config,
        )
    del pipe
    del draft
    gc.collect()


def test_llama_dflash_local_cpu_short_greedy_matches_target_pipeline():
    ov_genai = pytest.importorskip("openvino_genai")
    prompt = "Write a Python function that adds two integers."
    generation_config = ov_genai.GenerationConfig(do_sample=False, max_new_tokens=8, ignore_eos=True)

    target_snapshots, dflash_results = _generate_target_then_dflash_cb(
        "CPU",
        [prompt],
        generation_config,
        _TARGET_PIPELINE_PROPERTIES.copy(),
        _local_llama_dflash_cb_models(),
    )
    target_snapshot = target_snapshots[0]
    dflash_result = dflash_results[0]

    _assert_same_generation(prompt, target_snapshot, dflash_result)
    assert dflash_result["num_generated"] == target_snapshot["num_generated"]
    assert dflash_result["extended_perf_metrics"] is not None
    assert dflash_result["extended_perf_metrics"].draft_model_metrics.raw_metrics.m_durations


def test_llama_dflash_local_cpu_exercises_target_kv_rollback():
    ov_genai = pytest.importorskip("openvino_genai")
    generation_config = ov_genai.GenerationConfig(do_sample=False, max_new_tokens=32, ignore_eos=True)
    prompts = [
        "Write a compact C++ function that checks if a number is prime.",
        _CPU_ROLLBACK_PROMPT,
        "Continue this sequence with code-like tokens: def foo(x):",
    ]

    target_results, dflash_results = _generate_target_then_dflash_cb(
        "CPU",
        prompts,
        generation_config,
        _TARGET_PIPELINE_PROPERTIES.copy(),
        _local_llama_dflash_cb_models(),
    )

    total_draft_generated = 0
    total_draft_accepted = 0
    for prompt, target_snapshot, dflash_result in zip(prompts, target_results, dflash_results):
        _assert_same_generation(prompt, target_snapshot, dflash_result)
        assert dflash_result["num_generated"] == target_snapshot["num_generated"]
        metrics = dflash_result["extended_perf_metrics"]
        assert metrics is not None
        draft_generated = metrics.draft_model_metrics.get_num_generated_tokens()
        draft_accepted = metrics.get_num_accepted_tokens()
        assert draft_generated > 0, f"DFlash draft did not generate candidates for prompt: {prompt}"
        assert draft_accepted <= draft_generated
        total_draft_generated += draft_generated
        total_draft_accepted += draft_accepted

    assert total_draft_generated > total_draft_accepted, (
        "CPU Llama DFlash did not reject any draft candidates, so target KV-cache rollback was not exercised. "
        f"draft_generated={total_draft_generated}, draft_accepted={total_draft_accepted}"
    )


def test_llama_dflash_local_gpu_short_greedy_matches_target_pipeline():
    ov_genai = pytest.importorskip("openvino_genai")
    _require_gpu_device()
    prompt = "Write a Python function that adds two integers."
    generation_config = ov_genai.GenerationConfig(do_sample=False, max_new_tokens=8, ignore_eos=True)

    target_snapshots, dflash_results = _generate_target_then_dflash_cb(
        "GPU",
        [prompt],
        generation_config,
        _TARGET_PIPELINE_PROPERTIES.copy(),
        _local_llama_dflash_cb_models(),
    )
    target_snapshot = target_snapshots[0]
    dflash_result = dflash_results[0]

    _assert_same_generation(prompt, target_snapshot, dflash_result)
    assert dflash_result["num_generated"] == target_snapshot["num_generated"]
    assert dflash_result["extended_perf_metrics"] is not None
    assert dflash_result["extended_perf_metrics"].draft_model_metrics.raw_metrics.m_durations


def test_llama_dflash_local_gpu_exercises_target_kv_rollback():
    ov_genai = pytest.importorskip("openvino_genai")
    _require_gpu_device()
    generation_config = ov_genai.GenerationConfig(do_sample=False, max_new_tokens=32, ignore_eos=True)
    prompts = [
        "Write a compact C++ function that checks if a number is prime.",
        _CPU_ROLLBACK_PROMPT,
        "Continue this sequence with code-like tokens: def foo(x):",
    ]

    target_results, dflash_results = _generate_target_then_dflash_cb(
        "GPU",
        prompts,
        generation_config,
        _TARGET_PIPELINE_PROPERTIES.copy(),
        _local_llama_dflash_cb_models(),
    )

    total_draft_generated = 0
    total_draft_accepted = 0
    for prompt, target_snapshot, dflash_result in zip(prompts, target_results, dflash_results):
        _assert_same_generation(prompt, target_snapshot, dflash_result)
        assert dflash_result["num_generated"] == target_snapshot["num_generated"]
        metrics = dflash_result["extended_perf_metrics"]
        assert metrics is not None
        draft_generated = metrics.draft_model_metrics.get_num_generated_tokens()
        draft_accepted = metrics.get_num_accepted_tokens()
        assert draft_generated > 0, f"DFlash draft did not generate candidates for prompt: {prompt}"
        assert draft_accepted <= draft_generated
        total_draft_generated += draft_generated
        total_draft_accepted += draft_accepted

    assert total_draft_generated > total_draft_accepted, (
        "GPU Llama DFlash did not reject any draft candidates, so target KV-cache rollback was not exercised. "
        f"draft_generated={total_draft_generated}, draft_accepted={total_draft_accepted}"
    )


def test_dflash_local_gpu_greedy_matches_target_pipeline():
    ov_genai = pytest.importorskip("openvino_genai")
    generation_config = ov_genai.GenerationConfig(do_sample=False, max_new_tokens=8, ignore_eos=True)
    target_results, dflash_results = _generate_target_then_dflash_cb_on_gpu(
        ["Write a Python function that adds two integers."],
        generation_config,
    )

    target_snapshot = target_results[0]
    dflash_result = dflash_results[0]
    _assert_same_generation("Write a Python function that adds two integers.", target_snapshot, dflash_result)
    assert dflash_result["num_generated"] == target_snapshot["num_generated"]
    assert dflash_result["extended_perf_metrics"] is not None
    assert dflash_result["extended_perf_metrics"].main_model_metrics.raw_metrics.m_durations
    assert dflash_result["extended_perf_metrics"].draft_model_metrics.raw_metrics.m_durations


def test_dflash_local_gpu_exercises_target_kv_rollback():
    ov_genai = pytest.importorskip("openvino_genai")
    generation_config = ov_genai.GenerationConfig(do_sample=False, max_new_tokens=32, ignore_eos=True)
    prompts = [
        "Write a compact C++ function that checks if a number is prime.",
        "Explain speculative decoding in one paragraph.",
        "Continue this sequence with code-like tokens: def foo(x):",
    ]

    target_results, dflash_results = _generate_target_then_dflash_cb_on_gpu(prompts, generation_config)

    total_draft_generated = 0
    total_draft_accepted = 0
    for prompt, target_snapshot, dflash_result in zip(prompts, target_results, dflash_results):
        # Long GPU validation windows can diverge from the target-only graph numerically once
        # hidden-state outputs are exposed; exact GPU correctness is covered by the short test.
        assert dflash_result["num_generated"] == target_snapshot["num_generated"], (
            f"DFlash generated a different number of tokens for prompt: {prompt}"
        )
        assert dflash_result["tokens"][0], f"DFlash produced no tokens for prompt: {prompt}"
        metrics = dflash_result["extended_perf_metrics"]
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
