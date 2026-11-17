# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import SAMPLES_CPP_DIR, SAMPLES_PY_DIR
from test_utils import run_sample
from utils.bagel_test_assets import prepare_tiny_bagel_ov_path
from utils.constants import get_ov_cache_converted_models_dir


@pytest.fixture(scope="module")
def bagel_ov_path() -> str:
    return str(prepare_tiny_bagel_ov_path(get_ov_cache_converted_models_dir()))


@pytest.fixture(scope="module")
def input_image(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("bagel") / "input.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
    return str(path)


def bagel_command(sample: str, model_dir: str, task: str, prompt: str, images: list[str]) -> list[str]:
    if sample == "cpp":
        command = [str(SAMPLES_CPP_DIR / "bagel"), model_dir, "CPU", task, prompt]
    else:
        command = [sys.executable, str(SAMPLES_PY_DIR / "bagel/bagel.py"), model_dir, "CPU", task, prompt]
    return command + images


@pytest.mark.bagel
@pytest.mark.samples
@pytest.mark.parametrize("sample", ["cpp", "py"])
@pytest.mark.parametrize(
    "task, prompt, num_images",
    [
        ("vqa", "Describe the image.", 1),
        ("vqa", "Compare the images.", 3),
        ("t2i", "A red bicycle", 0),
        ("edit", "Make it green", 1),
        ("edit", "Combine the images", 3),
    ],
)
def test_sample_bagel(
    bagel_ov_path: str, input_image: str, sample: str, task: str, prompt: str, num_images: int, tmp_path: Path
):
    result = run_sample(
        bagel_command(sample, bagel_ov_path, task, prompt, [input_image] * num_images), cwd=str(tmp_path)
    )
    # a random tiny model may answer with an empty string, so VQA is only checked to run
    if task != "vqa":
        assert list(tmp_path.glob("image*.bmp")), f"{task} wrote no image:\n{result.stdout}"
