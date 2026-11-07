# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Offline utility for manually regenerating the committed forced-aligner reference JSON.
# CI never runs this script; the reference test compares GenAI against the committed JSON
# rather than running the qwen_asr PyTorch reference implementation used here.

import json
import pathlib

import numpy as np
import torch
from huggingface_hub import snapshot_download
from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

MODEL_ID = "optimum-intel-internal-testing/tiny-random-qwen3-forced-aligner"
# (id, transcript, language); ids identify reference cases in pytest output.
CASES = [
    ("english", "how're you doing today?", "english"),
    ("mixed-english-chinese", "hello你好world", "english"),
    ("chinese", "你好世界。", "chinese"),
]


def forced_aligner_audio():
    # Must stay identical to utils/asr_utils/qwen3_asr.py::forced_aligner_audio(), which the test
    # feeds to GenAI; the committed reference is only valid for this exact input.
    rng = np.random.default_rng(0)
    return (rng.standard_normal(16000) * 0.01).astype(np.float32)


if __name__ == "__main__":
    aligner = Qwen3ForcedAligner.from_pretrained(snapshot_download(MODEL_ID), dtype=torch.float32)

    reference = []
    for case_id, transcript, language in CASES:
        result = aligner.align((forced_aligner_audio(), 16000), transcript, language)[0]
        reference.append(
            {
                "id": case_id,
                "transcript": transcript,
                "language": language,
                "words": [
                    {
                        "text": item.text,
                        "start_ts": round(item.start_time, 3),
                        "end_ts": round(item.end_time, 3),
                    }
                    for item in result
                ],
            }
        )

    out_path = pathlib.Path(__file__).parent / "tiny_random_qwen3_forced_aligner_reference.json"
    out_path.write_text(json.dumps(reference, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
