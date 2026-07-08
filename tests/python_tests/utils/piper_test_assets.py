# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Tiny-random Piper (VITS-family) ONNX/OpenVINO fixture generator for fast unit testing.

Piper voices are distributed as a single raw ONNX graph (opset 15, ai.onnx domain only,
no custom/contrib ops) with a fixed I/O contract:

    Input  "input"          int64   [1, num_phoneme_ids]
    Input  "input_lengths"  int64   [1]
    Input  "scales"         float32 [3]  == [noise_scale, length_scale, noise_w]
    Output "output"         float32 [1, time, 1, 1]

Unlike Kokoro/SpeechT5 there is no HuggingFace/optimum-cli export path for this
architecture (it is not a Transformers model), so this fixture builds a tiny ONNX graph
directly with `onnx.helper` instead of exporting from a PyTorch checkpoint. The graph is a
structural stand-in only (embedding lookup -> linear upsample to a token-length-dependent
waveform); it does not reproduce Piper's real VITS acoustic modelling, only its I/O
contract, so tests built on it validate GenAI plumbing/parsing, not audio quality.
"""

import io
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import onnx
import openvino as ov
from onnx import TensorProto, helper, numpy_helper

from .atomic_download import AtomicDownloadManager

logger = logging.getLogger(__name__)

TINY_PIPER_SEED = 2024
TINY_PIPER_SAMPLE_RATE = 22050
TINY_PIPER_HIDDEN_DIM = 8
# Number of upsampled audio samples generated per input phoneme-id, so output length scales
# with (and reveals bugs in) the phoneme-id sequence building logic under test.
TINY_PIPER_SAMPLES_PER_TOKEN = 256

# Minimal, single-Unicode-codepoint phoneme_id_map covering BOS/PAD/EOS plus a handful of
# IPA-like letters/punctuation, following the real Piper sidecar convention:
#   PAD = phoneme_id_map["_"][0] (0), BOS = phoneme_id_map["^"][0] (1), EOS = phoneme_id_map["$"][0] (2)
TINY_PHONEME_ID_MAP = {
    "_": [0],
    "^": [1],
    "$": [2],
    " ": [3],
    ".": [4],
    "a": [5],
    "b": [6],
    "e": [7],
    "h": [8],
    "l": [9],
    "o": [10],
    "t": [11],
    "s": [12],
}
TINY_VOCAB_SIZE = max(ids[0] for ids in TINY_PHONEME_ID_MAP.values()) + 1


def _build_tiny_piper_onnx_model() -> onnx.ModelProto:
    """
    Build a tiny ONNX graph matching Piper's exact I/O contract.

    Graph: Gather(embedding, input) -> ReduceMean(axis=1) -> Tile by
    (input_lengths[0] * TINY_PIPER_SAMPLES_PER_TOKEN) -> reshape to [1, time, 1, 1].
    `scales` is consumed (added elementwise, scaled by a tiny constant) purely to exercise
    the input being wired up correctly; it does not meaningfully affect output shape.
    """
    rng = np.random.default_rng(TINY_PIPER_SEED)
    embedding = rng.uniform(-0.1, 0.1, size=(TINY_VOCAB_SIZE, TINY_PIPER_HIDDEN_DIM)).astype(np.float32)

    input_ids = helper.make_tensor_value_info("input", TensorProto.INT64, [1, None])
    input_lengths = helper.make_tensor_value_info("input_lengths", TensorProto.INT64, [1])
    scales = helper.make_tensor_value_info("scales", TensorProto.FLOAT, [3])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, None, 1, 1])

    embedding_initializer = numpy_helper.from_array(embedding, name="embedding_table")

    nodes = [
        helper.make_node("Gather", ["embedding_table", "input"], ["gathered"], axis=0),
        helper.make_node("ReduceMean", ["gathered"], ["pooled"], axes=[1, 2], keepdims=0),
        helper.make_node("ReduceSum", ["scales"], ["scale_sum"], keepdims=1),
        helper.make_node("ReduceMean", ["pooled"], ["pooled_scalar"], axes=[0], keepdims=0),
        helper.make_node("Add", ["pooled_scalar", "scale_sum"], ["mixed_scalar"]),
        helper.make_node(
            "Mul",
            ["input_lengths", numpy_helper.from_array(np.array([TINY_PIPER_SAMPLES_PER_TOKEN], dtype=np.int64), "spt").name],
            ["time_len"],
        ),
        helper.make_node("Reshape", ["time_len", numpy_helper.from_array(np.array([1], dtype=np.int64), "one_shape").name], ["time_len_1d"]),
        helper.make_node("Expand", ["mixed_scalar", "time_len_1d"], ["expanded"]),
        helper.make_node(
            "Reshape",
            ["expanded", numpy_helper.from_array(np.array([1, -1, 1, 1], dtype=np.int64), "out_shape").name],
            ["output"],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "tiny_piper_vits",
        [input_ids, input_lengths, scales],
        [output],
        initializer=[
            embedding_initializer,
            numpy_helper.from_array(np.array([TINY_PIPER_SAMPLES_PER_TOKEN], dtype=np.int64), "spt"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "one_shape"),
            numpy_helper.from_array(np.array([1, -1, 1, 1], dtype=np.int64), "out_shape"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    onnx.checker.check_model(model)
    return model


def generate_tiny_piper_model(output_dir: Path, voice_name: str = "tiny-random-piper", espeak_voice: str = "en-us") -> Path:
    """Generate a tiny random Piper genai model directory (config.json + OpenVINO IR)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_model = _build_tiny_piper_onnx_model()
    onnx_bytes = io.BytesIO(onnx_model.SerializeToString())
    ov_model = ov.convert_model(onnx_bytes)
    ov.save_model(ov_model, str(output_dir / "openvino_model.xml"))

    config = {
        "architectures": ["PiperForTextToSpeech"],
        "phoneme_id_map": TINY_PHONEME_ID_MAP,
        "sample_rate": TINY_PIPER_SAMPLE_RATE,
        "language": espeak_voice,
        "noise_scale": 0.667,
        "length_scale": 1.0,
        "noise_w": 0.8,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)

    logger.info("Generated tiny Piper model at %s", output_dir)
    return output_dir


def prepare_tiny_piper_ov_path(converted_models_dir: Path) -> Path:
    """Prepare a tiny random Piper OpenVINO model directory in cache and return its path."""
    model_path = Path(converted_models_dir) / "tiny-random-piper-ov"
    manager = AtomicDownloadManager(model_path)
    manager.execute(lambda temp_path: generate_tiny_piper_model(temp_path / "model"))

    nested = model_path / "model"
    if nested.exists() and (nested / "config.json").exists():
        if not (model_path / "config.json").exists():
            for item in nested.iterdir():
                shutil.move(str(item), str(model_path / item.name))
        shutil.rmtree(nested, ignore_errors=True)

    return model_path
