# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic GGUF multimodal comparison against the optimum-intel export of the same checkpoint.

The GGUF path loads its models in the export's layout while using GGUF media geometry and
token limits. Export with `optimum-cli export openvino --task image-text-to-text`,
including the tokenizer and the checkpoint's processor_config.json. Muse Glimmer GGUF files collapse
the two-frame patch kernel, so its video is expected to differ.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as genai
from validate_gguf_mmproj import Tokens

TEXTS = ["The weather today is.\nNext line.", "I'm sure they'll say it's DONE'S", "Numbers 1234567 and 3.14159",
         "Ünïcödé café — Привет мир! 你好，世界。", "  leading   spaces\t\ttabs\r\n\r\nwindows",
         "def f(x):\n    return x**2  # comment", "emoji 😀👍🏽"]


def generate(pipe, prompt, **media):
    streamer = Tokens()
    pipe.generate(prompt, max_new_tokens=20, do_sample=False, streamer=streamer, **media)
    return streamer.tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", type=Path)
    parser.add_argument("mmproj", type=Path)
    parser.add_argument("export", type=Path, help="optimum-intel export directory of the same checkpoint")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--attention-backend", choices=("SDPA", "PA"), default="SDPA")
    args = parser.parse_args()
    properties = dict(ATTENTION_BACKEND=args.attention_backend, INFERENCE_PRECISION_HINT="f32")
    ir = genai.VLMPipeline(str(args.export), "CPU", **properties)
    gguf = genai.VLMPipeline(str(args.language), "CPU", mmproj_path=str(args.mmproj), **properties)

    report = {"language": str(args.language), "mmproj": str(args.mmproj), "export": str(args.export),
              "openvino_version": ov.get_version(), "genai_version": genai.__version__,
              "tokenization": [], "generation": []}
    for text in TEXTS:
        expected, actual = (p.get_tokenizer().encode(text, add_special_tokens=False).input_ids.data.tolist()[0]
                            for p in (ir, gguf))
        report["tokenization"].append({"text": text, "passed": expected == actual, "ir": expected, "gguf": actual})

    rng = np.random.default_rng(0)
    cases = [("text", "What is 2 plus 2?", {}),
             ("image", "<ov_genai_image_0>Describe the image.",
              {"images": [ov.Tensor(rng.integers(0, 256, (1, 300, 420, 3), dtype=np.uint8))]}),
             ("two_images", "<ov_genai_image_0><ov_genai_image_1>Compare the images.",
              {"images": [ov.Tensor(rng.integers(0, 256, (1, 224, 224, 3), dtype=np.uint8)),
                          ov.Tensor(rng.integers(0, 256, (1, 160, 320, 3), dtype=np.uint8))]})]
    if args.video:
        cases.append(("video", "<ov_genai_video_0>Describe the video.",
                      {"videos": [ov.Tensor(rng.integers(0, 256, (4, 224, 224, 3), dtype=np.uint8))]}))
    for name, prompt, media in cases:
        expected, actual = generate(ir, prompt, **media), generate(gguf, prompt, **media)
        matching = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), min(len(expected), len(actual)))
        report["generation"].append({"modality": name, "passed": expected == actual, "matching_prefix": matching,
                                     "ir": expected, "gguf": actual})
        print(name, "MATCH" if expected == actual else f"DIFF after {matching} tokens", flush=True)

    report["passed"] = all(c["passed"] for c in report["tokenization"] + report["generation"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
