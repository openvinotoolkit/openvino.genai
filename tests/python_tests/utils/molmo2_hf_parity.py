# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compares the Molmo2 language model exported by genai-cli with the original Hugging Face decoder.

Usage (from tests/python_tests, so that utils/tokenizers.py does not shadow the tokenizers package)::

    python -m utils.molmo2_hf_parity <checkpoint dir> <OpenVINO model dir>

The Molmo2 remote code needs the transformers version the native exporter pins, so test_vlm_molmo2 runs this script in
a uv overlay providing it. Both models get the same input embeddings: a prefill with text and image tokens (image tokens
attend to each other bidirectionally) followed by cached decoding steps. Exits with an error if the logits differ.
"""

import sys

import numpy as np
import openvino as ov
import torch
from transformers import AutoModelForImageTextToText

# Positions of the image tokens (token_type_ids == 1) in the prefill: two runs, separated by text tokens.
_PREFILL_TOKEN_TYPES = [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0]
_DECODE_STEPS = 2
_ATOL = 1e-4


def main(checkpoint: str, ov_dir: str) -> None:
    torch.manual_seed(0)
    hf_model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, trust_remote_code=True, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    # f32 everywhere: the CPU plugin compresses the KV cache by default, which alone changes the decoding logits.
    ov_model = ov.Core().compile_model(
        f"{ov_dir}/openvino_language_model.xml",
        "CPU",
        {"INFERENCE_PRECISION_HINT": "f32", "KV_CACHE_PRECISION": "f32"},
    )
    request = ov_model.create_infer_request()
    request.reset_state()

    num_tokens = len(_PREFILL_TOKEN_TYPES) + _DECODE_STEPS
    vocab_size = hf_model.config.text_config.vocab_size
    with torch.no_grad():
        embeds = hf_model.model.transformer.wte(torch.randint(0, vocab_size, (1, num_tokens)))

    past_key_values, prefill_len = None, len(_PREFILL_TOKEN_TYPES)
    chunks = [(0, prefill_len, _PREFILL_TOKEN_TYPES)]
    chunks += [(position, position + 1, [0]) for position in range(prefill_len, num_tokens)]
    for start, end, token_types in chunks:
        inputs_embeds = embeds[:, start:end]
        attention_mask = torch.ones(1, end, dtype=torch.int64)
        position_ids = torch.arange(start, end).unsqueeze(0)
        token_type_ids = torch.tensor([token_types])
        with torch.no_grad():
            hf_out = hf_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = hf_out.past_key_values
        ov_logits = request.infer(
            {
                "inputs_embeds": inputs_embeds.numpy(),
                "attention_mask": attention_mask.numpy(),
                "position_ids": position_ids.numpy(),
                "token_type_ids": token_type_ids.numpy(),
                "beam_idx": np.zeros(1, dtype=np.int32),
            }
        )["logits"]
        max_diff = float(np.abs(ov_logits - hf_out.logits.numpy()).max())
        print(f"tokens [{start}, {end}): max abs logits difference {max_diff:.2e}")
        if max_diff > _ATOL:
            sys.exit(f"The OpenVINO logits of tokens [{start}, {end}) differ from Hugging Face by {max_diff}")


if __name__ == "__main__":
    main(*sys.argv[1:])
