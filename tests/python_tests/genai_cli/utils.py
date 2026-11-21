# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import openvino as ov
from huggingface_hub import snapshot_download
from openvino_genai import VLMPipeline
from openvino_genai._cli.exporters import get_native_exporter

from utils.export import export_with_genai_cli, native_genai_cli_model_type
from utils.network import retry_request

# Tiny checkpoints of the model types genai-cli exports natively.
SUPPORTED_ARCHITECTURES = {
    "molmo2": "optimum-intel-internal-testing/tiny-random-molmo2",
}

# The OpenVINO GenAI pipeline and the IRs of every task the native exporters produce.
TASK_PIPELINES = {
    "image-text-to-text": VLMPipeline,
}
TASK_SUBMODELS = {
    "image-text-to-text": (
        "openvino_language_model.xml",
        "openvino_text_embeddings_model.xml",
        "openvino_vision_embeddings_model.xml",
    ),
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


_COMPRESSED_ELEMENT_TYPES = {
    "i8": "int8",
    "u8": "int8",
    "i4": "int4",
    "u4": "int4",
    "nf4": "nf4",
    "f4e2m1": "f4e2m1",
    "f8e8m0": "f8e8m0",
    "f8e4m3": "f8e4m3",
}


def num_compressed_weight_nodes(ov_model: ov.Model) -> dict[str, int]:
    """Counts nodes with low-precision outputs per type, as optimum-intel's
    tests/openvino/utils_tests.get_num_quantized_nodes."""
    counts = {}
    for op in ov_model.get_ops():
        for output in op.outputs():
            element_type = _COMPRESSED_ELEMENT_TYPES.get(output.get_element_type().get_type_name())
            if element_type is not None:
                counts[element_type] = counts.get(element_type, 0) + 1
    return counts


def generates_text(output: Path, task: str) -> bool:
    result = TASK_PIPELINES[task](output, "CPU").generate("Hello", max_new_tokens=3, do_sample=False)
    return len(result.texts) == 1
