# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export tests of the genai-cli export tool"""

import json
import shutil
from pathlib import Path

import openvino as ov
import pytest
from openvino_genai import Tokenizer
from openvino_genai._cli.exporters import _import_exporter_modules, get_native_exporter
from openvino_genai._cli.exporters.base import registered_exporters

from genai_cli.utils import (
    EXPORTS,
    SUPPORTED_ARCHITECTURES,
    TASK_PIPELINES,
    TASK_SUBMODELS,
    download,
    export,
    generates_text,
    num_compressed_weight_nodes,
)

# Submodel holding the KV cache, which GenAI pipelines need to be stateful.
TASK_STATEFUL_SUBMODEL = {
    "image-text-to-text": "openvino_language_model.xml",
}


def _has_constants_of_type(ov_model: ov.Model, element_type: ov.Type) -> bool:
    return any(op.get_type_name() == "Constant" and op.get_element_type() == element_type for op in ov_model.get_ops())


@pytest.fixture(scope="module", params=EXPORTS, ids=[f"{model_type}-{task}" for model_type, task in EXPORTS])
def exported_model(request, tmp_path_factory) -> tuple[str, str, Path, Path]:
    model_type, task = request.param
    model_dir = download(model_type)
    output = tmp_path_factory.mktemp(f"{model_type}-{task}")
    export(model_dir, output, model_type, ["--task", task])
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
        assert num_compressed_weight_nodes(ov.Core().read_model(output / submodel)) == {}

    assert generates_text(output, task)


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
    # A writable local copy of the checkpoint: shutil.copyfile doesn't preserve the permissions of the Hub cache files.
    model_dir = tmp_path / "checkpoint"
    shutil.copytree(download(model_type), model_dir, copy_function=shutil.copyfile)
    generation_config_path = model_dir / "generation_config.json"
    generation_config = json.loads(generation_config_path.read_text())
    generation_config.update(top_k=42, do_sample=True)
    generation_config_path.write_text(json.dumps(generation_config))

    output = tmp_path / "exported"
    export(model_dir, output, model_type, ["--task", task])
    exported_generation_config = json.loads((output / "generation_config.json").read_text())
    assert exported_generation_config["top_k"] == 42
    assert exported_generation_config["do_sample"] is True

    config = TASK_PIPELINES[task](output, "CPU").get_generation_config()
    assert config.top_k == 42
    assert config.do_sample


@pytest.mark.parametrize("model_type, task", EXPORTS)
def test_export_fp16_model(tmp_path, model_type, task):
    export(download(model_type), tmp_path, model_type, ["--task", task, "--weight-format", "fp16"])
    stateful_model = ov.Core().read_model(tmp_path / TASK_STATEFUL_SUBMODEL[task])
    assert _has_constants_of_type(stateful_model, ov.Type.f16)
    assert stateful_model.get_sinks()

    assert generates_text(tmp_path, task)


@pytest.mark.parametrize("model_type, task", EXPORTS)
def test_export_with_disable_convert_tokenizer(tmp_path, model_type, task):
    export(download(model_type), tmp_path, model_type, ["--task", task, "--disable-convert-tokenizer"])
    for submodel in TASK_SUBMODELS[task]:
        assert (tmp_path / submodel).is_file()
    assert not (tmp_path / "openvino_tokenizer.xml").exists()
    assert not (tmp_path / "openvino_detokenizer.xml").exists()


@pytest.mark.parametrize("model_type, task", EXPORTS)
def test_export_with_custom_variant(tmp_path, model_type, task):
    # A local checkpoint that only has the weights of the `custom` variant.
    model_dir = tmp_path / "checkpoint"
    shutil.copytree(download(model_type), model_dir, copy_function=shutil.copyfile)
    (model_dir / "model.safetensors").rename(model_dir / "model.custom.safetensors")

    output = tmp_path / "exported"
    export(model_dir, output, model_type, ["--task", task, "--variant", "custom"])
    assert generates_text(output, task)
