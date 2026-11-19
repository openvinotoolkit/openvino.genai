# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Checks that OpenVINO applies the expected transformations to the models exported by genai-cli."""

import re
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

from genai_cli.utils import EXPORTS, SUPPORTED_ARCHITECTURES, TASK_PIPELINES, download, export

# Expected transformations per model type, separated by stage:
#   "convert" - transformations applied while genai-cli exports the model
#   "compile" - CPU plugin transformations applied while the GenAI pipeline compiles the model
EXPECTED_TRANSFORMATIONS = {
    "molmo2": {
        "convert": [
            "SDPAFusion",
            "MakeStateful",
            "CommonFusions",
        ],
        "compile": [
            "RMSFusion",
            "RoPEFusion",
            "RoPEFusionGPTNEOX",
            "SDPASubgraphFusion",
            "CausalMaskPreprocessFusion",
            "ConvertMatMulToFC",
            "FullyConnectedBiasFusion",
            "ConvertToPowerStatic",
            "ConvertToSwishCPU",
            "Snippets",
            "Tokenization",
        ],
    },
}

# Transformations that adapt the stateful language model to the attention backend of GenAI pipelines.
ATTENTION_BACKEND_TRANSFORMATIONS = {
    "PA": ["StateManagementPattern"],
    "SDPA": ["StatefulSDPAFusion"],
}

# With OV_ENABLE_PROFILE_PASS, OpenVINO logs every transformation as "<name> <time>ms +" when it changed the model
# and as "<name> <time>ms -" when it did not.
_PASS_LOG_LINE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s+\d+ms\s*([+-])\s*$")


def _check_transformations(log: str, expected: list[str]) -> None:
    applied, not_applied = set(), set()
    for line in log.splitlines():
        match = _PASS_LOG_LINE.search(line)
        if match:
            (applied if match.group(2) == "+" else not_applied).add(match.group(1))
    errors = []
    if missing := [name for name in expected if name not in applied and name in not_applied]:
        errors.append(f"not applied: {', '.join(missing)}")
    if missing := [name for name in expected if name not in applied and name not in not_applied]:
        errors.append(f"not found: {', '.join(missing)}")
    assert not errors, "; ".join(errors)


@pytest.fixture(scope="module", params=EXPORTS, ids=[f"{model_type}-{task}" for model_type, task in EXPORTS])
def profiled_export(request, tmp_path_factory) -> tuple[str, str, Path, str]:
    model_type, task = request.param
    output = tmp_path_factory.mktemp(f"{model_type}-{task}")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("OV_ENABLE_PROFILE_PASS", "1")
        log = export(download(model_type), output, model_type, ["--task", task])
    return model_type, task, output, log


def test_every_native_exporter_has_expected_transformations():
    assert set(EXPECTED_TRANSFORMATIONS) == set(SUPPORTED_ARCHITECTURES)


def test_convert_transformations(profiled_export):
    model_type, _, _, log = profiled_export
    _check_transformations(log, EXPECTED_TRANSFORMATIONS[model_type]["convert"])


@pytest.mark.parametrize("attention_backend", ATTENTION_BACKEND_TRANSFORMATIONS)
def test_compile_transformations(profiled_export, monkeypatch, attention_backend):
    model_type, task, output, _ = profiled_export
    # A subprocess, OpenVINO writes the pass log from C++.
    code = (
        "import sys, openvino_genai\n"
        f"openvino_genai.{TASK_PIPELINES[task].__name__}(sys.argv[1], 'CPU', ATTENTION_BACKEND=sys.argv[2])\n"
    )
    monkeypatch.setenv("OV_ENABLE_PROFILE_PASS", "1")
    result = subprocess.run(
        [sys.executable, "-c", code, str(output), attention_backend],
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    _check_transformations(
        result.stdout,
        EXPECTED_TRANSFORMATIONS[model_type]["compile"] + ATTENTION_BACKEND_TRANSFORMATIONS[attention_backend],
    )
