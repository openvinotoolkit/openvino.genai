# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
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


def test_dflash_local_greedy_matches_target_pipeline():
    target_model_path, draft_model_path = _local_dflash_models()
    ov_genai = pytest.importorskip("openvino_genai")
    prompt = "Write a Python function that adds two integers."
    generation_config = ov_genai.GenerationConfig(do_sample=False, max_new_tokens=8, ignore_eos=True)

    target_pipe = ov_genai.LLMPipeline(target_model_path, "CPU", {})
    draft = ov_genai.draft_model(draft_model_path, "CPU")
    dflash_pipe = ov_genai.LLMPipeline(target_model_path, "CPU", {}, draft_model=draft)

    target_result = target_pipe.generate([prompt], generation_config)
    dflash_result = dflash_pipe.generate([prompt], generation_config)

    assert dflash_result.texts == target_result.texts
    assert dflash_result.perf_metrics.get_num_generated_tokens() == target_result.perf_metrics.get_num_generated_tokens()


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
