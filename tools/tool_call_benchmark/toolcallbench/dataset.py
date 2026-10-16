# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset loading with integrity pinning for the tool_call_benchmark."""

import hashlib
import json
import os

DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "coding_agent_v1.jsonl")
DATASET_SHA256 = "0937084f88d736d13761eb86e4f85560cf998b6cb334083cb118e715a0ede02d"
DATASET_VERSION = "coding_agent_v1"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(path=None):
    """Load the frozen benchmark dataset.

    :param path: dataset JSONL path; None selects the packaged default,
        whose sha256 is verified against DATASET_SHA256.
    :return: (meta dict, list of case dicts)
    :raises ValueError: if the packaged dataset fails the sha256 check.
    """
    path = path or DEFAULT_DATASET_PATH
    if path == DEFAULT_DATASET_PATH:
        digest = _sha256(path)
        if digest != DATASET_SHA256:
            raise ValueError(
                f"dataset integrity check failed: sha256 {digest} != pinned {DATASET_SHA256}")
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    meta, cases = rows[0]["_meta"], rows[1:]
    return meta, cases
