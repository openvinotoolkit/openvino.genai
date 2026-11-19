# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Weight compression tests of the genai-cli export tool"""

import openvino as ov
import pytest

from genai_cli.utils import TASK_SUBMODELS, download, export, generates_text, num_compressed_weight_nodes

# Submodel that gets the compression requested with --weight-format.
TASK_COMPRESSED_SUBMODEL = {
    "image-text-to-text": "openvino_language_model.xml",
}
# Low-precision weight nodes of the other submodels, which get int8_sym weights whatever compression is requested.
INT8_SYM_SUBMODELS_WEIGHT_NODES = {
    "molmo2": {
        "openvino_text_embeddings_model.xml": {"int8": 1},
        "openvino_vision_embeddings_model.xml": {"int8": 26},
    },
}

# Compression options, the expected number of low-precision weight nodes (compressed weights, zero points and scales)
# of the compressed submodel, and the expected parameters NNCF stores in its rt_info.
WEIGHT_COMPRESSION_CONFIGURATIONS = [
    (
        "molmo2",
        "image-text-to-text",
        ["--weight-format", "int8"],
        {"int8": 44},
        {"mode": "int8_asym"},
    ),
    (
        "molmo2",
        "image-text-to-text",
        ["--weight-format", "int4"],
        {"int8": 2, "int4": 42},
        {"mode": "int4_asym", "ratio": "1.0", "group_size": "128"},
    ),
    (
        "molmo2",
        "image-text-to-text",
        ["--weight-format", "int4", "--ratio", "0.8", "--group-size", "-1"],
        {"int8": 8, "int4": 36},
        {"mode": "int4_asym", "ratio": "0.8", "group_size": "-1"},
    ),
    (
        "molmo2",
        "image-text-to-text",
        ["--weight-format", "int4", "--group-size", "16", "--all-layers"],
        {"int4": 44},
        {"mode": "int4_asym", "group_size": "16", "all_layers": "True"},
    ),
    (
        "molmo2",
        "image-text-to-text",
        ["--weight-format", "int4", "--group-size", "16", "--ratio", "0.5", "--backup-precision", "none"],
        {"int4": 16},
        {"mode": "int4_asym", "ratio": "0.5", "backup_mode": "none"},
    ),
    # The hidden size of the tiny model is not divisible by 48.
    (
        "molmo2",
        "image-text-to-text",
        ["--weight-format", "int4", "--sym", "--group-size", "48", "--group-size-fallback", "adjust"],
        {"int8": 2, "int4": 21},
        {"mode": "int4_sym", "group_size": "48"},
    ),
    (
        "molmo2",
        "image-text-to-text",
        ["--weight-format", "int4", "--group-size", "16", "--awq"],
        {"int8": 2, "int4": 42},
        {"mode": "int4_asym", "group_size": "16", "awq": "True"},
    ),
    (
        "molmo2",
        "image-text-to-text",
        ["--weight-format", "mxfp4"],
        {"f4e2m1": 21, "f8e8m0": 22, "f8e4m3": 1},
        {"mode": "mxfp4"},
    ),
    (
        "molmo2",
        "image-text-to-text",
        ["--weight-format", "nf4"],
        {"int8": 2, "nf4": 21},
        {"mode": "nf4"},
    ),
    (
        "molmo2",
        "image-text-to-text",
        ["--weight-format", "cb4"],
        {"int8": 23, "int4": 21, "f8e4m3": 21},
        {"mode": "cb4"},
    ),
]


@pytest.mark.parametrize(
    "model_type, task, options, expected_num_weight_nodes, expected_rt_info",
    WEIGHT_COMPRESSION_CONFIGURATIONS,
    ids=[f"{model_type}-{' '.join(options)}" for model_type, _, options, _, _ in WEIGHT_COMPRESSION_CONFIGURATIONS],
)
def test_weight_compression(tmp_path, model_type, task, options, expected_num_weight_nodes, expected_rt_info):
    export(download(model_type), tmp_path, model_type, ["--task", task, *options])
    num_weight_nodes = {
        submodel: num_compressed_weight_nodes(ov.Core().read_model(tmp_path / submodel))
        for submodel in TASK_SUBMODELS[task]
    }
    compressed_submodel = TASK_COMPRESSED_SUBMODEL[task]
    assert num_weight_nodes == {
        compressed_submodel: expected_num_weight_nodes,
        **INT8_SYM_SUBMODELS_WEIGHT_NODES[model_type],
    }

    rt_info = ov.Core().read_model(tmp_path / compressed_submodel).get_rt_info()["nncf"]["weight_compression"]
    assert {key: rt_info[key].value for key in expected_rt_info} == expected_rt_info

    assert generates_text(tmp_path, task)
