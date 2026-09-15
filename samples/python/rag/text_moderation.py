#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

import numpy as np
import openvino_genai

# A user turn is scored by the query_* heads, an assistant turn by the other two. Each head is
# paired with the config.json map naming its classes, e.g. {"0": "Safe", "1": "Unsafe"}.
USER_HEADS = (("query_risk_level_logits", "query_risk_level_map"), ("query_category_logits", "query_category_map"))
ASSISTANT_HEADS = (("risk_level_logits", "response_risk_level_map"), ("category_logits", "response_category_map"))


def print_verdict(title, scores, heads, label_maps):
    verdict = []
    for name, map_name in heads:
        logits = scores[name].data[0, -1]
        winner = int(np.argmax(logits))
        probability = 1 / np.exp(logits - logits[winner]).sum()
        label = label_maps.get(map_name, {}).get(str(winner), winner)
        verdict.append(f"{label:<14} p={probability:.3f}")
    print(f"{title:<10} {'   '.join(verdict)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("prompt")
    parser.add_argument("response", nargs="?", default=None)
    args = parser.parse_args()

    device = "CPU"  # GPU and NPU can be used as well

    label_maps = json.loads((Path(args.model_dir) / "config.json").read_text())
    pipeline = openvino_genai.TextEmbeddingPipeline(args.model_dir, device)
    tokenizer = openvino_genai.Tokenizer(args.model_dir)

    prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], False)
    print_verdict("prompt", pipeline.score([prompt]), USER_HEADS, label_maps)

    if args.response is None:
        return

    if pipeline.is_stateful():
        # the model keeps the KV cache of everything scored so far, so the response can be moderated
        # token by token as it is generated, e.g. from an LLMPipeline streamer
        pipeline.reset_state()
        pipeline.score_next(tokenizer.encode(prompt).input_ids)
        for token in tokenizer.encode(args.response, add_special_tokens=False).input_ids.data[0]:
            print_verdict("  token", pipeline.score_next([int(token)]), ASSISTANT_HEADS, label_maps)
    else:
        print_verdict("response", pipeline.score([prompt + args.response]), ASSISTANT_HEADS, label_maps)


if "__main__" == __name__:
    main()
