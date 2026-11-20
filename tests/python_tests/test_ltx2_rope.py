# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import openvino as ov
import openvino_genai as genai
import pytest
import torch
from diffusers.models.transformers.transformer_ltx2 import LTX2AudioVideoRotaryPosEmbed
from diffusers.pipelines.ltx2.connectors import LTX2RotaryPosEmbed1d
from openvino import opset13 as ops

ROPE_NAMES = ("video", "audio", "video_cross_attn", "audio_cross_attn")
CONFIG = {
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "audio_num_attention_heads": 2,
    "audio_attention_head_dim": 8,
    "audio_cross_attention_dim": 16,
    "in_channels": 4,
    "pos_embed_max_pos": 12,
    "audio_pos_embed_max_pos": 24,
    "base_height": 1024,
    "base_width": 512,
}


def _parameter(name, shape, dtype=np.float32):
    parameter = ops.parameter(shape, dtype, name=name)
    parameter.output(0).get_tensor().set_names({name})
    return parameter


def _save_probe(path, parameters, outputs, config):
    path.mkdir(exist_ok=True)
    results = []
    for name, output in outputs.items():
        result = ops.result(output)
        result.output(0).get_tensor().set_names({name})
        results.append(result)
    ov.save_model(ov.Model(results, parameters), path / "openvino_model.xml", compress_to_fp16=False)
    (path / "config.json").write_text(json.dumps(config))
    return path


def _transformer_probe(path, rope_type, double_precision=True, contract="external", timestep_rank=2):
    parameters = [
        _parameter("hidden_states", [-1, -1, 4]),
        _parameter("audio_hidden_states", [-1, -1, 4]),
        _parameter("timestep", [-1] if timestep_rank == 1 else [-1, -1]),
    ]
    config = dict(CONFIG, rope_type=rope_type, rope_double_precision=double_precision)
    if contract in ("legacy", "missing_coords"):
        video = _parameter("video_coords", [-1, 3, -1, 2])
        audio = _parameter("audio_coords", [-1, 1, -1, 2])
        parameters.extend([video, audio])
        outputs = {
            "out_sample": ops.add(video, ops.constant(0.0, np.float32)),
            "audio_out_sample": ops.add(audio, ops.constant(0.0, np.float32)),
        }
        if contract == "missing_coords":
            parameters.remove(audio)
            outputs["audio_out_sample"] = ops.add(parameters[1], ops.constant(0.0, np.float32))
    else:
        rotary = {}
        for name in ROPE_NAMES:
            dim = 32 if name == "video" else 16
            shape = [-1, 2, -1, dim // 4] if rope_type == "split" else [-1, -1, dim]
            for suffix in ("cos", "sin"):
                parameter = _parameter(f"{name}_rotary_{suffix}", shape)
                parameters.append(parameter)
                rotary[name, suffix] = parameter
        outputs = {
            output: ops.concat([ops.reshape(rotary[name, suffix], [-1], False) for name in ROPE_NAMES], 0)
            for output, suffix in (("out_sample", "cos"), ("audio_out_sample", "sin"))
        }
        if contract == "incomplete":
            parameters.remove(rotary["audio", "sin"])
            outputs = {
                "out_sample": ops.add(parameters[0], ops.constant(0.0, np.float32)),
                "audio_out_sample": ops.add(parameters[1], ops.constant(0.0, np.float32)),
            }
        elif contract == "mixed":
            parameters.extend([_parameter("video_coords", [-1, 3, -1, 2]), _parameter("audio_coords", [-1, 1, -1, 2])])
    return _save_probe(path, parameters, outputs, config)


def _coords(batch, dimensions, length):
    # Include large angles, nonzero interval widths, and different positions in each batch.
    starts = np.linspace(0, 40, batch * dimensions * length, dtype=np.float32).reshape(batch, dimensions, length)
    return np.stack((starts, starts + np.float32(0.25)), axis=-1)


@pytest.mark.parametrize("rope_type", ["split", "interleaved"])
@pytest.mark.parametrize("double_precision", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_transformer_rope_reference(tmp_path, rope_type, double_precision, batch):
    path = _transformer_probe(tmp_path / "transformer", rope_type, double_precision)
    transformer = genai.LTX2VideoTransformer3DModel(path)
    transformer.reshape(batch, 9, 32, 32, 3)
    transformer.compile("CPU", INFERENCE_PRECISION_HINT="f32")
    video, audio = _coords(batch, 3, 2), _coords(batch, 1, 3)
    transformer.set_rope_coords(ov.Tensor(video), ov.Tensor(audio))
    actual = transformer.infer(
        ov.Tensor(np.zeros((batch, 2, 4), np.float32)), ov.Tensor(np.zeros((batch, 3, 4), np.float32)), 500.0
    )
    expected = [[], []]
    for name in ROPE_NAMES:
        cross = "cross_attn" in name
        coords = video if name.startswith("video") else audio
        if cross:
            coords = coords[:, :1]
        reference = LTX2AudioVideoRotaryPosEmbed(
            dim=32 if name == "video" else 16,
            num_attention_heads=2,
            modality="video" if name == "video" else "audio",
            double_precision=double_precision,
            rope_type=rope_type,
            base_num_frames=24 if cross or name == "audio" else 12,
            base_height=1024,
            base_width=512,
        )
        for index, values in enumerate(reference(torch.from_numpy(coords))):
            expected[index].append(values.numpy().reshape(-1))
    for output, values in zip(actual, expected):
        np.testing.assert_allclose(output.data, np.concatenate(values), rtol=0, atol=2e-6 if double_precision else 3e-3)


def _connector_probe(path, rope_type, double_precision=True, contract="external"):
    hidden = _parameter("text_encoder_hidden_states", [-1, -1, 4])
    mask = _parameter("attention_mask", [-1, -1])
    parameters = [hidden, mask]
    config = {"rope_type": rope_type, "rope_double_precision": double_precision}
    cos, sin = [], []
    for name, heads, head_dim in (("video_connector", 2, 16), ("audio_connector", 3, 11)):
        if rope_type == "split":
            heads, head_dim = 2, 16
        dim = heads * head_dim
        config.update({name + "_num_attention_heads": heads, name + "_attention_head_dim": head_dim})
        if contract == "legacy":
            continue
        shape = [-1, heads, -1, dim // 2 // heads] if rope_type == "split" else [-1, -1, dim]
        for suffix, target in (("cos", cos), ("sin", sin)):
            parameter = _parameter(f"{name}_rotary_{suffix}", shape)
            parameters.append(parameter)
            target.append(ops.reshape(parameter, [-1], False))
    if contract == "incomplete":
        parameters.pop()
    outputs = {
        "video_text_embedding": ops.add(hidden, ops.constant(0.0, np.float32)),
        "audio_text_embedding": ops.add(hidden, ops.constant(0.0, np.float32)),
        "connector_attention_mask": ops.add(mask, ops.constant(0.0, np.float32)),
    }
    if contract == "external":
        outputs["video_text_embedding"] = ops.concat(cos, 0)
        outputs["audio_text_embedding"] = ops.concat(sin, 0)
    return _save_probe(path, parameters, outputs, config), config


@pytest.mark.parametrize("rope_type", ["split", "interleaved"])
@pytest.mark.parametrize("double_precision", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_connector_rope_reference(tmp_path, rope_type, double_precision, batch):
    path, config = _connector_probe(tmp_path / "connectors", rope_type, double_precision)
    connectors = genai.LTX2TextConnectors(path)
    connectors.reshape(batch)
    connectors.compile("CPU", INFERENCE_PRECISION_HINT="f32")
    output = connectors.infer(
        ov.Tensor(np.zeros((batch, 7, 4), np.float32)), ov.Tensor(np.ones((batch, 7), np.float32))
    )
    expected = [[], []]
    for name in ("video_connector", "audio_connector"):
        heads = config[name + "_num_attention_heads"]
        dim = heads * config[name + "_attention_head_dim"]
        reference = LTX2RotaryPosEmbed1d(
            dim, num_attention_heads=heads, rope_type=rope_type, double_precision=double_precision
        )
        for index, values in enumerate(reference(batch, 7, "cpu")):
            expected[index].append(values.numpy().reshape(-1))
    for actual, values in zip(output[:2], expected):
        np.testing.assert_allclose(actual.data, np.concatenate(values), rtol=0, atol=2e-6 if double_precision else 3e-3)


@pytest.mark.parametrize("timestep_rank", [1, 2])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_legacy_transformer_coordinates(tmp_path, timestep_rank, static_shapes):
    path = _transformer_probe(tmp_path / "transformer", "interleaved", contract="legacy", timestep_rank=timestep_rank)
    transformer = genai.LTX2VideoTransformer3DModel(path)
    assert transformer.has_input("video_coords")
    assert not transformer.has_input("video_rotary_cos")
    if static_shapes:
        transformer.reshape(2, 9, 32, 32, 3)
    transformer.compile("CPU")
    assert transformer.has_input("video_coords")
    assert not transformer.has_input("video_rotary_cos")
    video, audio = _coords(2, 3, 2), _coords(2, 1, 3)
    transformer.set_rope_coords(ov.Tensor(video), ov.Tensor(audio))
    actual = transformer.infer(
        ov.Tensor(np.zeros((2, 2, 4), np.float32)), ov.Tensor(np.zeros((2, 3, 4), np.float32)), 500.0
    )
    np.testing.assert_array_equal(actual[0].data, video)
    np.testing.assert_array_equal(actual[1].data, audio)


@pytest.mark.parametrize("contract", ["external", "legacy"])
@pytest.mark.parametrize("timestep_rank", [1, 2])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_timestep_input_contract(tmp_path, contract, timestep_rank, static_shapes):
    path = _transformer_probe(tmp_path / "transformer", "interleaved", contract=contract, timestep_rank=timestep_rank)
    graph = ov.Core().read_model(path / "openvino_model.xml")
    timestep = graph.input("timestep").get_node()
    outputs = {}
    for name in ("out_sample", "audio_out_sample"):
        result = ops.result(ops.add(timestep, ops.constant(0.0, np.float32)))
        result.output(0).get_tensor().set_names({name})
        outputs[name] = result
    ov.save_model(
        ov.Model(list(outputs.values()), graph.get_parameters()), path / "openvino_model.xml", compress_to_fp16=False
    )
    transformer = genai.LTX2VideoTransformer3DModel(path)
    if static_shapes:
        transformer.reshape(2, 9, 32, 32, 3)
    transformer.compile("CPU")
    transformer.set_rope_coords(ov.Tensor(_coords(2, 3, 2)), ov.Tensor(_coords(2, 1, 3)))
    output = transformer.infer(
        ov.Tensor(np.zeros((2, 2, 4), np.float32)), ov.Tensor(np.zeros((2, 3, 4), np.float32)), 500.0
    )
    shape = (2,) if timestep_rank == 1 else (2, 1 if contract == "external" else 2)
    np.testing.assert_array_equal(output[0].data, np.full(shape, 500.0, np.float32))


def test_legacy_connectors(tmp_path):
    path, _ = _connector_probe(tmp_path / "connectors", "interleaved", contract="legacy")
    connectors = genai.LTX2TextConnectors(path)
    connectors.reshape(2)
    connectors.compile("CPU")
    hidden = np.arange(56, dtype=np.float32).reshape(2, 7, 4)
    output = connectors.infer(ov.Tensor(hidden), ov.Tensor(np.ones((2, 7), np.float32)))
    np.testing.assert_array_equal(output[0].data, hidden)
    np.testing.assert_array_equal(output[1].data, hidden)


@pytest.mark.parametrize(
    "component, contract",
    [
        ("transformer", "incomplete"),
        ("transformer", "mixed"),
        ("transformer", "missing_coords"),
        ("connectors", "incomplete"),
    ],
)
def test_invalid_rotary_contract_rejected(tmp_path, component, contract):
    if component == "transformer":
        path = _transformer_probe(tmp_path / component, "split", contract=contract)
        model = genai.LTX2VideoTransformer3DModel(path)
    else:
        path, _ = _connector_probe(tmp_path / component, "split", contract=contract)
        model = genai.LTX2TextConnectors(path)
    with pytest.raises(RuntimeError, match="rotary"):
        model.compile("CPU")
