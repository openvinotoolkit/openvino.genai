# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import struct
from pathlib import Path

import numpy as np


def write_les_predictor(path: Path, hidden_size: int = 8, num_experts: int = 4, decoder_layers=(2,)) -> Path:
    """Writes a random LES predictor file (expert_predictors.<layer>.* tensors in safetensors format).

    Devices without LES support (e.g. CPU) only read the file and ignore the predictor, so the file does not need to
    match the model: it checks that a tool passes the les_predictor property to the pipeline.
    """
    rng = np.random.default_rng(0)
    tensors = {}
    for layer in decoder_layers:
        prefix = f"expert_predictors.{layer}."
        tensors[prefix + "weight"] = rng.standard_normal((num_experts, hidden_size), dtype=np.float32)
        tensors[prefix + "previous_weight"] = rng.standard_normal((num_experts, hidden_size), dtype=np.float32)
        tensors[prefix + "selection_bias"] = rng.standard_normal(num_experts, dtype=np.float32)
        tensors[prefix + "input_mode_code"] = np.array(2, dtype=np.int64)  # current + previous token

    header, data = {}, b""
    for name, tensor in tensors.items():
        raw = tensor.tobytes()
        header[name] = {
            "dtype": "F32" if tensor.dtype == np.float32 else "I64",
            "shape": list(tensor.shape),
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    header_bytes = json.dumps(header).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)
    path = Path(path)
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + data)
    return path
