# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import sys
import os
import shutil
import numpy as np

from pathlib import Path

import openvino as ov


# NOTE:
# Using regex to edit XML is the minimal method to generate the model with custom op "MyAdd" (`type="Add"` -> `type="MyAdd"`, `version="extension"`).
# We intentionally avoid `from_pretrained`/`save_pretrained` because that path adds extra
# conversion/serialization code and broadens test scope beyond custom-op dispatch.
def replace_ir_add_with_myadd(ir_xml_path: Path, target_type_name: str = "MyAdd") -> None:
    import re

    xml_text = ir_xml_path.read_text(encoding="utf-8")

    layer_pattern = re.compile(r'(<layer\b[^>]*\btype="Add"[^>]*>)(.*?)(</layer>)', re.DOTALL)
    match = None
    for candidate in layer_pattern.finditer(xml_text):
        layer_body = candidate.group(2)
        input_match = re.search(r"<input\b[^>]*>(.*?)</input>", layer_body, re.DOTALL)
        if input_match is None:
            continue

        input_ports = re.findall(r"<port\b[^>]*>.*?</port>", input_match.group(1), re.DOTALL)
        if len(input_ports) != 2:
            continue

        second_input_port = input_ports[1]
        if re.search(r'\bprecision="FP32"', second_input_port) is None:
            continue

        input_dims = re.findall(r"<dim>\s*(\d+)\s*</dim>", second_input_port)
        if not input_dims:
            continue
        if any(dim != "1" for dim in input_dims):
            continue

        match = candidate
        break

    assert match is not None, (
        f"No suitable IR Add layer found to replace with {target_type_name}. "
        "Expected: type='Add', exactly two inputs, and second input with precision='FP32' and all dims equal to 1"
    )

    layer_tag = match.group(1)
    updated_tag = re.sub(r'\btype="Add"', f'type="{target_type_name}"', layer_tag, count=1)
    assert updated_tag != layer_tag, "Selected IR layer does not have type='Add'"

    if re.search(r'\bversion="[^"]*"', updated_tag):
        updated_tag = re.sub(r'\bversion="[^"]*"', 'version="extension"', updated_tag, count=1)
    else:
        updated_tag = updated_tag[:-1] + ' version="extension">'

    xml_text = xml_text[: match.start(1)] + updated_tag + xml_text[match.end(1) :]
    ir_xml_path.write_text(xml_text, encoding="utf-8")


def get_ir_xml_path(model_dir: Path) -> Path:
    default_ir_xml_path = model_dir / "openvino_model.xml"
    if default_ir_xml_path.exists():
        return default_ir_xml_path

    language_model_ir_xml_path = model_dir / "openvino_language_model.xml"
    if language_model_ir_xml_path.exists():
        return language_model_ir_xml_path

    raise FileNotFoundError(
        f"IR XML was not found: {default_ir_xml_path} or {language_model_ir_xml_path}"
    )


def assert_ir_contains_op_type(model_path: Path | str, op_type: str) -> None:
    import re

    model_dir = Path(model_path)
    ir_xml_path = get_ir_xml_path(model_dir)

    xml_text = ir_xml_path.read_text(encoding="utf-8")
    pattern = re.compile(rf'<layer\b[^>]*\btype="{re.escape(op_type)}"[^>]*>')
    assert pattern.search(xml_text) is not None, f"IR does not contain op type '{op_type}' in {ir_xml_path}"


def get_extension_model(model_path: str, temp_dir: Path, op_name: str = "MyAdd") -> Path:
    source_path = Path(model_path)
    extension_suffix = "extension" if op_name == "MyAdd" else f"{op_name.lower()}_extension"
    extension_path = temp_dir / f"{source_path.name}_{extension_suffix}"
    default_ir_xml_path = extension_path / "openvino_model.xml"
    language_model_ir_xml_path = extension_path / "openvino_language_model.xml"
    if default_ir_xml_path.exists() or language_model_ir_xml_path.exists():
        assert_ir_contains_op_type(extension_path, op_name)
        return extension_path

    if not source_path.exists():
        raise FileNotFoundError(f"Model path was not found: {source_path}")

    shutil.copytree(source_path, extension_path)
    ir_xml_path = get_ir_xml_path(extension_path)
    replace_ir_add_with_myadd(ir_xml_path, target_type_name=op_name)
    assert_ir_contains_op_type(extension_path, op_name)

    assert op_name.encode("utf-8") in ir_xml_path.read_bytes(), (
        f"Custom op '{op_name}' was not injected into OpenVINO IR"
    )
    return extension_path


def get_extension_lib_path():
    extension_name = "openvino_custom_add_extension"
    if sys.platform == "win32":
        suffixes = [".dll"]
    elif sys.platform == "darwin":
        suffixes = [".dylib", ".so"]
    else:
        suffixes = [".so"]

    file_names = []
    for suffix in suffixes:
        file_names.append(f"lib{extension_name}{suffix}")
        file_names.append(f"{extension_name}{suffix}")

    extension_lib_path_env = os.getenv("EXTENSION_LIB_PATH")
    if extension_lib_path_env:
        env_path = Path(extension_lib_path_env)
        if env_path.is_file():
            return env_path
        if env_path.exists():
            for file_name in file_names:
                matches = sorted(env_path.rglob(file_name))
                if matches:
                    return matches[0]

    workspace_root = Path(__file__).resolve().parents[3]
    build_dir = workspace_root / "build"
    if not build_dir.exists():
        raise FileNotFoundError(f"Build directory was not found: {build_dir}")

    for file_name in file_names:
        matches = sorted(build_dir.rglob(file_name))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not find compiled custom extension '{extension_name}'. "
        f"Searched EXTENSION_LIB_PATH='{os.getenv('EXTENSION_LIB_PATH')}' and build directory '{build_dir}'. "
    )


class CustomAdd(ov.Op):
    class_type_info = ov.DiscreteTypeInfo("CustomAdd", "extension")
    evaluate_calls = 0

    def __init__(self, inputs=None, **attrs):
        super().__init__(self, inputs)
        self._attrs = attrs

    def validate_and_infer_types(self):
        self.set_output_type(0, self.get_input_element_type(0), self.get_input_partial_shape(0))

    def clone_with_new_inputs(self, new_inputs):
        return CustomAdd(new_inputs, **self._attrs)

    def visit_attributes(self, visitor):
        visitor.on_attributes(self._attrs)
        return True

    def has_evaluate(self):
        return True

    def evaluate(self, outputs, inputs):
        type(self).evaluate_calls += 1
        lhs = np.array(inputs[0].data, copy=False)
        rhs = np.array(inputs[1].data, copy=False)
        outputs[0].shape = inputs[0].shape
        np.add(lhs, rhs, out=np.array(outputs[0].data, copy=False))
        return True

    def get_type_info(self):
        return CustomAdd.class_type_info
