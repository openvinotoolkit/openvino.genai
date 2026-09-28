# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import sys
from pathlib import Path

import pytest

from conftest import convert_model, run_wwb
from test_cli_speechgen import get_overall_score


QWEN3_VOICE_DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
QWEN3_CUSTOM_VOICE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
QWEN3_BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
QWEN3_OV_CONFIG = '{"KV_CACHE_PRECISION": "f32"}'


def _run_qwen3_case(
    *,
    model_id: str,
    case_name: str,
    tmp_root: Path,
    extra_args: list[str] | None = None,
) -> dict[str, object]:
    extra_args = extra_args or []

    gt_file = tmp_root / f"{case_name}_gt.csv"
    model_path = convert_model(model_id)
    base_output_dir = tmp_root / f"{case_name}_base_genai"
    optimum_output_dir = tmp_root / f"{case_name}_optimum"
    genai_output_dir = tmp_root / f"{case_name}_genai"

    base_args = [
        "--base-model",
        model_id,
        "--num-samples",
        "1",
        "--gt-data",
        gt_file,
        "--device",
        "CPU",
        "--model-type",
        "speech-generation",
        "--hf",
        "--output",
        base_output_dir,
        *extra_args,
    ]
    run_wwb(base_args)

    optimum_args = [
        "--target-model",
        model_path,
        "--num-samples",
        "1",
        "--gt-data",
        gt_file,
        "--device",
        "CPU",
        "--model-type",
        "speech-generation",
        "--ov-config",
        QWEN3_OV_CONFIG,
        "--output",
        optimum_output_dir,
        *extra_args,
    ]
    optimum_output = run_wwb(optimum_args)

    genai_args = [
        "--target-model",
        model_path,
        "--num-samples",
        "1",
        "--gt-data",
        gt_file,
        "--device",
        "CPU",
        "--model-type",
        "speech-generation",
        "--ov-config",
        QWEN3_OV_CONFIG,
        "--genai",
        "--output",
        genai_output_dir,
        *extra_args,
    ]
    genai_output = run_wwb(genai_args)

    optimum_score = get_overall_score(optimum_output)
    genai_score = get_overall_score(genai_output)

    assert math.isfinite(optimum_score), "Optimum score must be finite"
    assert math.isfinite(genai_score), "GenAI score must be finite"

    return {
        "gt_file": gt_file,
        "base_output_dir": base_output_dir,
        "optimum_score": optimum_score,
        "genai_score": genai_score,
    }


@pytest.fixture(scope="module")
def qwen3_voice_design_artifacts(tmp_path_factory):
    tmp_root = tmp_path_factory.mktemp("qwen3_voice_design")
    result = _run_qwen3_case(
        model_id=QWEN3_VOICE_DESIGN_MODEL_ID,
        case_name="voice_design",
        tmp_root=tmp_root,
        extra_args=["--speech-instruct", "warm, calm narration"],
    )

    # Speech-generation reference files are saved under <dirname(gt-data)>/reference.
    # Reuse the VoiceDesign reference clip for Qwen3 Base tests.
    ref_audio = Path(result["gt_file"]).parent / "reference" / "0.wav"
    assert ref_audio.exists(), f"Reference audio was not generated: {ref_audio}"

    return {
        **result,
        "ref_audio": str(ref_audio),
    }


def test_tts_qwen3_voice_design(qwen3_voice_design_artifacts):
    assert qwen3_voice_design_artifacts["optimum_score"] >= 0.95
    assert qwen3_voice_design_artifacts["genai_score"] >= 0.95


def test_tts_qwen3_custom_voice_no_instruct(tmp_path):
    result = _run_qwen3_case(
        model_id=QWEN3_CUSTOM_VOICE_MODEL_ID,
        case_name="custom_voice",
        tmp_root=tmp_path,
        extra_args=[
            "--speech-voice",
            "Vivian",
        ],
    )

    assert result["optimum_score"] >= 0.95
    assert result["genai_score"] >= 0.95


def test_tts_qwen3_custom_voice_with_instruct(tmp_path):
    result = _run_qwen3_case(
        model_id=QWEN3_CUSTOM_VOICE_MODEL_ID,
        case_name="custom_voice",
        tmp_root=tmp_path,
        extra_args=[
            "--speech-voice",
            "Vivian",
            "--speech-instruct",
            "friendly and clear",
        ],
    )

    assert result["optimum_score"] >= 0.95
    assert result["genai_score"] >= 0.95


def test_tts_qwen3_base_icl_mode(qwen3_voice_design_artifacts, tmp_path):
    result = _run_qwen3_case(
        model_id=QWEN3_BASE_MODEL_ID,
        case_name="base_icl",
        tmp_root=tmp_path,
        extra_args=[
            "--speech-ref-audio",
            qwen3_voice_design_artifacts["ref_audio"],
            "--speech-ref-text",
            "Hello and welcome to the OpenVINO speech generation benchmark.",
        ],
    )

    assert result["optimum_score"] >= 0.95
    assert result["genai_score"] >= 0.95


def test_tts_qwen3_base_xvector_mode(qwen3_voice_design_artifacts, tmp_path):
    result = _run_qwen3_case(
        model_id=QWEN3_BASE_MODEL_ID,
        case_name="base_xvector",
        tmp_root=tmp_path,
        extra_args=[
            "--speech-ref-audio",
            qwen3_voice_design_artifacts["ref_audio"],
        ],
    )

    assert result["optimum_score"] >= 0.95
    assert result["genai_score"] >= 0.95
