# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import subprocess  # nosec B404
import sys

import pytest


@pytest.mark.vlm
@pytest.mark.samples
@pytest.mark.parametrize(
    "option, count, error",
    [
        ("-n", "0", "--num_iter must be positive"),
        ("-n", "-1", "--num_iter must be positive"),
        ("-N", "-1", "--num_warmup must be non-negative"),
    ],
)
def test_python_benchmark_invalid_iteration_counts(option, count, error):
    samples_dir = Path(os.environ.get("SAMPLES_PY_DIR", Path(__file__).resolve().parents[3] / "samples/python"))
    benchmark_script = samples_dir / "visual_language_chat/benchmark_vlm.py"
    result = subprocess.run(
        [sys.executable, benchmark_script, "-m", "unused_model", option, count],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert error in result.stderr
