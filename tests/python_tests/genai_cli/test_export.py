# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export tests of the genai-cli export tool"""

import json
import shutil
from pathlib import Path

import openvino as ov
import pytest
from huggingface_hub import snapshot_download
from openvino_genai import Tokenizer, VLMPipeline
from openvino_genai._cli.exporters import _import_exporter_modules, get_native_exporter
from openvino_genai._cli.exporters.base import registered_exporters

from genai_cli.utils import SUPPORTED_ARCHITECTURES
from utils.export import export_with_genai_cli, native_genai_cli_model_type
from utils.network import retry_request

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
# Submodel holding the KV cache, which GenAI pipelines need to be stateful.
TASK_STATEFUL_SUBMODEL = {
    "image-text-to-text": "openvino_language_model.xml",
}

EXPORTS = [
    (model_type, task)
    for model_type in SUPPORTED_ARCHITECTURES
    for task in get_native_exporter(model_type, None).TASKS
]

# Number of int8 / int4 weight nodes (compressed weights and zero points) per submodel for --weight-format options.
# The requested compression applies to the language model, the other submodels get int8_sym weights.
COMPRESSION_CONFIGURATIONS = [
    (
        "molmo2",
        "image-text-to-text",
        ["--weight-format", "int8"],
        {
            "openvino_language_model.xml": {"int8": 44, "int4": 0},
            "openvino_text_embeddings_model.xml": {"int8": 1, "int4": 0},
            "openvino_vision_embeddings_model.xml": {"int8": 26, "int4": 0},
        },
    ),
    (
        "molmo2",
        "image-text-to-text",
        ["--weight-format", "int8", "--sym"],
        {
            "openvino_language_model.xml": {"int8": 22, "int4": 0},
            "openvino_text_embeddings_model.xml": {"int8": 1, "int4": 0},
            "openvino_vision_embeddings_model.xml": {"int8": 26, "int4": 0},
        },
    ),
    (
        "molmo2",
        "image-text-to-text",
        ["--weight-format", "int4"],
        {
            "openvino_language_model.xml": {"int8": 2, "int4": 42},
            "openvino_text_embeddings_model.xml": {"int8": 1, "int4": 0},
            "openvino_vision_embeddings_model.xml": {"int8": 26, "int4": 0},
        },
    ),
    (
        "molmo2",
        "image-text-to-text",
        ["--weight-format", "int4", "--sym", "--group-size", "16", "--ratio", "0.5"],
        {
            "openvino_language_model.xml": {"int8": 28, "int4": 8},
            "openvino_text_embeddings_model.xml": {"int8": 1, "int4": 0},
            "openvino_vision_embeddings_model.xml": {"int8": 26, "int4": 0},
        },
    ),
]

_COMPRESSED_ELEMENT_TYPES = {"i8": "int8", "u8": "int8", "i4": "int4", "u4": "int4"}


def _download(model_type: str) -> Path:
    return Path(retry_request(lambda: snapshot_download(SUPPORTED_ARCHITECTURES[model_type])))


def _export(model_dir: Path, output: Path, model_type: str, options: list[str] = ()) -> None:
    # The checkpoint must go through the native exporter, not through the optimum-intel fallback.
    assert native_genai_cli_model_type(str(model_dir)) == model_type
    remote_code = ["--trust-remote-code"] if get_native_exporter(model_type, None).REQUIRES_REMOTE_CODE else []
    export_with_genai_cli(model_dir, output, [*remote_code, *options])


def _has_constants_of_type(ov_model: ov.Model, element_type: ov.Type) -> bool:
    return any(
        op.get_type_name() == "Constant" and op.get_element_type() == element_type for op in ov_model.get_ops()
    )


def _num_compressed_weight_nodes(ov_model: ov.Model) -> dict[str, int]:
    """Counts nodes with int8 / int4 outputs, as optimum-intel's tests/openvino/utils_tests.get_num_quantized_nodes."""
    counts = {"int8": 0, "int4": 0}
    for op in ov_model.get_ops():
        for output in op.outputs():
            element_type = _COMPRESSED_ELEMENT_TYPES.get(output.get_element_type().get_type_name())
            if element_type is not None:
                counts[element_type] += 1
    return counts


def _generates_text(output: Path, task: str) -> bool:
    result = TASK_PIPELINES[task](output, "CPU").generate("Hello", max_new_tokens=3, do_sample=False)
    return len(result.texts) == 1


@pytest.fixture(scope="module", params=EXPORTS, ids=[f"{model_type}-{task}" for model_type, task in EXPORTS])
def exported_model(request, tmp_path_factory) -> tuple[str, str, Path, Path]:
    model_type, task = request.param
    model_dir = _download(model_type)
    output = tmp_path_factory.mktemp(f"{model_type}-{task}")
    _export(model_dir, output, model_type, ["--task", task])
    return model_type, task, model_dir, output


def test_every_native_exporter_is_tested():
    _import_exporter_modules()
    assert set(registered_exporters()) == set(SUPPORTED_ARCHITECTURES)
    for model_type in SUPPORTED_ARCHITECTURES:
        assert set(get_native_exporter(model_type, None).TASKS) <= set(TASK_PIPELINES)


def test_export(exported_model):
    model_type, task, _, output = exported_model
    for submodel in (*TASK_SUBMODELS[task], "openvino_tokenizer.xml", "openvino_detokenizer.xml"):
        assert (output / submodel).is_file(), f"{submodel} was not created by the {model_type} export"

    stateful_model = ov.Core().read_model(output / TASK_STATEFUL_SUBMODEL[task])
    assert stateful_model.get_sinks(), "the KV cache must be stored in the model states"
    input_names = {name for model_input in stateful_model.inputs for name in model_input.get_names()}
    assert "beam_idx" in input_names
    assert not any(name.startswith("past_key_values") for name in input_names)
    # Without --weight-format, the tiny models keep their fp32 weights.
    assert not _has_constants_of_type(stateful_model, ov.Type.f16)
    for submodel in TASK_SUBMODELS[task]:
        assert _num_compressed_weight_nodes(ov.Core().read_model(output / submodel)) == {"int8": 0, "int4": 0}

    assert _generates_text(output, task)


def test_exported_tokenizer(exported_model):
    _, _, model_dir, output = exported_model
    tokenizer = Tokenizer(output)
    # The processor's chat template, which also lays out the images, takes precedence over the tokenizer's one.
    assert tokenizer.get_original_chat_template() == (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
    # Batches are padded on the left, as GenAI pipelines expect.
    attention_mask = tokenizer.encode(["Hi", "Hello there, how are you"]).attention_mask.data
    assert attention_mask[0, 0] == 0 and attention_mask[0, -1] == 1


@pytest.mark.parametrize("model_type, task", EXPORTS)
def test_export_with_custom_gen_config(tmp_path, model_type, task):
    # A local copy of the checkpoint, its Hub cache files are read-only.
    model_dir = tmp_path / "checkpoint"
    shutil.copytree(_download(model_type), model_dir)
    generation_config_path = model_dir / "generation_config.json"
    generation_config = json.loads(generation_config_path.read_text())
    generation_config.update(top_k=42, do_sample=True)
    generation_config_path.write_text(json.dumps(generation_config))

    output = tmp_path / "exported"
    _export(model_dir, output, model_type, ["--task", task])
    exported_generation_config = json.loads((output / "generation_config.json").read_text())
    assert exported_generation_config["top_k"] == 42
    assert exported_generation_config["do_sample"] is True

    config = TASK_PIPELINES[task](output, "CPU").get_generation_config()
    assert config.top_k == 42
    assert config.do_sample


@pytest.mark.parametrize("model_type, task", EXPORTS)
def test_export_fp16_model(tmp_path, model_type, task):
    _export(_download(model_type), tmp_path, model_type, ["--task", task, "--weight-format", "fp16"])
    stateful_model = ov.Core().read_model(tmp_path / TASK_STATEFUL_SUBMODEL[task])
    assert _has_constants_of_type(stateful_model, ov.Type.f16)
    assert stateful_model.get_sinks()

    assert _generates_text(tmp_path, task)


@pytest.mark.parametrize("model_type, task, options, expected_num_weight_nodes", COMPRESSION_CONFIGURATIONS)
def test_export_with_weight_compression(tmp_path, model_type, task, options, expected_num_weight_nodes):
    _export(_download(model_type), tmp_path, model_type, ["--task", task, *options])
    num_weight_nodes = {
        submodel: _num_compressed_weight_nodes(ov.Core().read_model(tmp_path / submodel))
        for submodel in TASK_SUBMODELS[task]
    }
    assert num_weight_nodes == expected_num_weight_nodes
    assert _generates_text(tmp_path, task)


@pytest.mark.parametrize("model_type, task", EXPORTS)
def test_export_with_disable_convert_tokenizer(tmp_path, model_type, task):
    _export(_download(model_type), tmp_path, model_type, ["--task", task, "--disable-convert-tokenizer"])
    for submodel in TASK_SUBMODELS[task]:
        assert (tmp_path / submodel).is_file()
    assert not (tmp_path / "openvino_tokenizer.xml").exists()
    assert not (tmp_path / "openvino_detokenizer.xml").exists()


@pytest.mark.parametrize("model_type, task", EXPORTS)
def test_export_with_custom_variant(tmp_path, model_type, task):
    # A local checkpoint that only has the weights of the `custom` variant.
    model_dir = tmp_path / "checkpoint"
    shutil.copytree(_download(model_type), model_dir)
    (model_dir / "model.safetensors").rename(model_dir / "model.custom.safetensors")

    output = tmp_path / "exported"
    _export(model_dir, output, model_type, ["--task", task, "--variant", "custom"])
    assert _generates_text(output, task)
