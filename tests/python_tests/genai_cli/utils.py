# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from huggingface_hub import snapshot_download
from openvino_genai import VLMPipeline
from openvino_genai._cli.exporters import get_native_exporter

from utils.export import export_with_genai_cli, native_genai_cli_model_type
from utils.network import retry_request

# Tiny checkpoints of the model types genai-cli exports natively.
SUPPORTED_ARCHITECTURES = {
    "molmo2": "optimum-intel-internal-testing/tiny-random-molmo2",
}

# The OpenVINO GenAI pipeline of every task the native exporters produce.
TASK_PIPELINES = {
    "image-text-to-text": VLMPipeline,
}

EXPORTS = [
    (model_type, task)
    for model_type in SUPPORTED_ARCHITECTURES
    for task in get_native_exporter(model_type, None).TASKS
]


def download(model_type: str) -> Path:
    return Path(retry_request(lambda: snapshot_download(SUPPORTED_ARCHITECTURES[model_type])))


def export(model_dir: Path, output: Path, model_type: str, options: list[str] = ()) -> str:
    """Exports with genai-cli and returns its output."""
    # The checkpoint must go through the native exporter, not through the optimum-intel fallback.
    assert native_genai_cli_model_type(str(model_dir)) == model_type
    remote_code = ["--trust-remote-code"] if get_native_exporter(model_type, None).REQUIRES_REMOTE_CODE else []
    return export_with_genai_cli(model_dir, output, [*remote_code, *options])
