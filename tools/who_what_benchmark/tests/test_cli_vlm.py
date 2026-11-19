# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from transformers import __version__ as transformers_version
from packaging.version import Version

from conftest import convert_model, run_wwb
from test_cli_image import get_similarity


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_test(model_id, model_type, optimum_threshold, genai_threshold, tmp_path, dataset=None, genai_reference=False):
    if sys.platform == 'darwin':
        pytest.xfail("Ticket 173169")

    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = convert_model(model_id)
    dataset_args = ["--dataset", str(dataset)] if dataset is not None else []

    # Collect reference
    run_wwb(
        [
            "--base-model",
            MODEL_PATH if genai_reference else model_id,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai" if genai_reference else "--hf",
        ]
        + dataset_args
    )

    # test Optimum
    output = run_wwb(
        [
            "--target-model",
            MODEL_PATH,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
        ]
        + dataset_args
    )
    if optimum_threshold is not None:
        similarity = get_similarity(output)
        assert similarity >= optimum_threshold

    # test GenAI
    output = run_wwb(
        [
            "--target-model",
            MODEL_PATH,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
            "--output",
            tmp_path,
        ]
        + dataset_args
    )
    if genai_threshold is not None:
        similarity = get_similarity(output)
        assert similarity >= genai_threshold

    # test w/o models
    run_wwb([
        "--target-data",
        tmp_path / "target.csv",
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
        "--genai",
    ])


def run_test_with_lora(
    model_id: str,
    model_type: str,
    lora_repo_id: str,
    lora_cache_subdir: str,
    hf_alpha: float,
    genai_alpha: float,
    tmp_path,
    *,
    genai_threshold: float,
):
    if sys.platform == "darwin":
        pytest.xfail("Ticket 173169")

    gt_file = tmp_path / "gt.csv"
    model_path = convert_model(model_id)

    from ov_utils import get_ov_cache_dir
    from ov_utils import download_hf_files_to_cache

    lora_filenames = ["adapter_model.safetensors", "adapter_config.json"]
    lora_cache_dir = get_ov_cache_dir() / "test_data" / lora_cache_subdir
    lora_adapter_dir = download_hf_files_to_cache(lora_repo_id, lora_cache_dir, lora_filenames)
    lora_adapter_file = lora_adapter_dir / "adapter_model.safetensors"
    assert lora_adapter_file.exists(), f"LoRA adapter wasn't downloaded: {lora_adapter_file}"

    # 1) Generate GT using HF + LoRA.
    run_wwb(
        [
            "--base-model",
            model_id,
            "--num-samples",
            "1",
            "--gt-data",
            gt_file,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--hf",
            "--adapters",
            str(lora_adapter_dir),
            "--alphas",
            str(hf_alpha),
            "--max_new_tokens",
            "32",
        ]
    )
    assert gt_file.exists(), f"GT wasn't generated: {gt_file}"

    # 2) Target: GenAI + LoRA
    outputs_genai = tmp_path / "genai_lora"
    out_genai = run_wwb(
        [
            "--target-model",
            model_path,
            "--num-samples",
            "1",
            "--gt-data",
            gt_file,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
            "--max_new_tokens",
            "32",
            "--output",
            outputs_genai,
            "--adapters",
            str(lora_adapter_file),
            "--alphas",
            str(genai_alpha),
        ]
    )

    assert (outputs_genai / "target.csv").exists()
    assert (outputs_genai / "metrics_per_question.csv").exists()
    assert (outputs_genai / "metrics.csv").exists()
    assert "Metrics for model" in out_genai
    similarity_genai = get_similarity(out_genai)

    assert similarity_genai >= genai_threshold


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [
        ("optimum-intel-internal-testing/tiny-random-llava", "visual-text"),
    ],
)
def test_vlm_basic(model_id, model_type, tmp_path):
    from PIL import Image

    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (32, 32), "white").save(image_path)
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame({"prompts": ["What animal is this?"], "images": [str(image_path)]}).to_csv(dataset, index=False)

    run_test(model_id, model_type, None, None, tmp_path, dataset=dataset, genai_reference=True)


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [
        ("optimum-intel-internal-testing/tiny-random-llava", "visual-text-chat"),
        ("optimum-intel-internal-testing/tiny-random-mistral3", "visual-text-chat"),
    ],
)
def test_vlm_chat(model_id, model_type, tmp_path):
    expected_prompts = ["Describe the animal in this photograph.", "What color are the cabinets behind the animal?"]
    dataset = tmp_path / "chat.json"
    dataset.write_text(
        json.dumps(
            [
                [
                    {"text": expected_prompts[0], "images": ["http://images.cocodataset.org/val2017/000000329219.jpg"]},
                    {"text": expected_prompts[1], "images": []},
                ]
            ]
        ),
        encoding="utf-8",
    )
    run_test(model_id, model_type, None, None, tmp_path, dataset=dataset)
    for csv_path in (tmp_path / "gt.csv", tmp_path / "target.csv"):
        row = pd.read_csv(csv_path, keep_default_na=False).iloc[0]
        assert json.loads(Path(row["prompts"]).read_text(encoding="utf-8")) == expected_prompts


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [
        ("optimum-intel-internal-testing/tiny-random-llava", "visual-text-only"),
    ],
)
def test_vlm_text_only(model_id, model_type, tmp_path):
    run_test(model_id, model_type, None, None, tmp_path)


@pytest.mark.transformers_higher_v5_1
@pytest.mark.skipif(
    Version(transformers_version) < Version("5.1"),
    reason="transformers 5.0.x reads an unset use_sliding_window in the Qwen3-Omni talker config; fixed in 5.1",
)
@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [
        ("optimum-intel-internal-testing/tiny-random-qwen3-omni", "visual-text"),
    ],
)
def test_vlm_qwen3_omni(model_id, model_type, tmp_path):
    run_test(model_id, model_type, None, None, tmp_path)


@pytest.mark.nanollava
@pytest.mark.parametrize(
    ("model_id", "model_type", "optimum_threshold", "genai_threshold"),
    [
        ("qnguyen3/nanoLLaVA", "visual-text", 0.99, 0.88),
    ],
)
def test_vlm_nanollava(model_id, model_type, optimum_threshold, genai_threshold, tmp_path):
    run_test(model_id, model_type, optimum_threshold, genai_threshold, tmp_path)


if Version(transformers_version) < Version("5.0.0"):
    # llava_next_video is not supported yet by optimum-intel 423b423 with transformers 5.0
    # videochat_flash_qwen is incompatible with transformers >= 5.0.0
    VISUAL_VIDEO_TEXT_MODELS = [
        ("optimum-intel-internal-testing/tiny-random-llava-next-video", "visual-video-text"),
        pytest.param(
            "optimum-intel-internal-testing/tiny-videochat-flash-qwen",
            "visual-video-text",
            marks=pytest.mark.xfail(
                sys.platform == "win32",
                reason="PyAV DLL load failure on Windows. Ticket CVS-183222",
                run=False,
            ),
        ),
    ]
else:
    VISUAL_VIDEO_TEXT_MODELS = [
        ("optimum-intel-internal-testing/tiny-random-qwen2vl", "visual-video-text"),
    ]


@pytest.mark.transformers_dependent(
    reason="llava_next_video is not supported yet by optimum-intel 423b423 with transformers 5.0"
)
@pytest.mark.parametrize(("model_id", "model_type"), VISUAL_VIDEO_TEXT_MODELS)
def test_vlm_video(model_id, model_type, tmp_path):
    run_test(model_id, model_type, 0.8, 0.8, tmp_path)


@pytest.mark.parametrize(
    (
        "model_id",
        "model_type",
        "lora_repo_id",
        "hf_alpha",
        "genai_alpha",
        "genai_threshold",
    ),
    [
        (
            "optimum-intel-internal-testing/tiny-random-qwen2vl",
            "visual-text",
            "likholat/tiny-random-qwen2vl-lora",
            1.0,
            2.0,
            0.99,
        ),
    ],
)
def test_vlm_lora(
    model_id,
    model_type,
    lora_repo_id,
    hf_alpha,
    genai_alpha,
    genai_threshold,
    tmp_path,
):
    run_test_with_lora(
        model_id=model_id,
        model_type=model_type,
        lora_repo_id=lora_repo_id,
        lora_cache_subdir="wwb_tiny_random_qwen2vl_lora",
        hf_alpha=hf_alpha,
        genai_alpha=genai_alpha,
        tmp_path=tmp_path,
        genai_threshold=genai_threshold,
    )
