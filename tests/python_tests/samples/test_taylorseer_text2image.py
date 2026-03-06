# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys
import numpy as np
from pathlib import Path
from PIL import Image

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample


def compare_images(image_path1: Path, image_path2: Path) -> bool:
    """Compare two images pixel by pixel for exact match."""
    with Image.open(image_path1) as img1, Image.open(image_path2) as img2:
        arr1 = np.array(img1)
        arr2 = np.array(img2)

        return arr1.shape == arr2.shape and np.array_equal(arr1, arr2)


class TestTaylorSeerText2Image:
    PROMPT = "cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting"

    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("tiny-random-flux", PROMPT, marks=pytest.mark.tiny_random_flux),
            pytest.param("stable-diffusion-3-tiny-random", PROMPT, marks=pytest.mark.stable_diffusion_3_tiny_random),
        ],
        indirect=["convert_model"],
    )
    def test_sample_taylorseer_text2image_default(self, convert_model, sample_args, tmp_path):
        """Test TaylorSeer text2image sample with default cache configuration."""
        py_dir = tmp_path / "python_output"
        cpp_dir = tmp_path / "cpp_output"
        py_dir.mkdir()
        cpp_dir.mkdir()

        # Run Python sample
        py_script = SAMPLES_PY_DIR / "image_generation/taylorseer_text2image.py"
        py_command = [sys.executable, py_script, convert_model, sample_args]
        run_sample(py_command, cwd=str(py_dir))

        # Run C++ sample
        cpp_sample = SAMPLES_CPP_DIR / "taylorseer_text2image"
        cpp_command = [cpp_sample, convert_model, sample_args]
        run_sample(cpp_command, cwd=str(cpp_dir))

        # Verify images exist and are identical
        py_baseline = py_dir / "taylorseer_baseline.bmp"
        py_cached = py_dir / "taylorseer.bmp"
        cpp_baseline = cpp_dir / "taylorseer_baseline.bmp"
        cpp_cached = cpp_dir / "taylorseer.bmp"

        assert py_baseline.exists(), f"Python baseline image not found: {py_baseline}"
        assert py_cached.exists(), f"Python cached image not found: {py_cached}"
        assert cpp_baseline.exists(), f"C++ baseline image not found: {cpp_baseline}"
        assert cpp_cached.exists(), f"C++ cached image not found: {cpp_cached}"

        # Compare baseline images (Python vs C++)
        assert compare_images(py_baseline, cpp_baseline), (
            "Baseline images from Python and C++ samples are not identical"
        )

        # Compare cached images (Python vs C++)
        assert compare_images(py_cached, cpp_cached), (
            "TaylorSeer cached images from Python and C++ samples are not identical"
        )
