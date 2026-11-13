#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import time
from collections import defaultdict

import numpy as np
import openvino_genai
from openvino import Tensor
from PIL import Image


def print_report(perf):
    print("compilation")
    print(f"  {'model / variant':<44}{'ms':>11}")
    for name, ms in perf.compile_ms:
        print(f"  {name:<44}{ms:>11.1f}")
    print(f"  {'total':<44}{sum(ms for _, ms in perf.compile_ms):>11.1f}")
    by_stage = defaultdict(list)
    for stage, ms in perf.infer_ms:
        by_stage[stage].append(ms)
    print("inference")
    print(f"  {'stage':<26}{'calls':>6}{'total ms':>11}{'mean ms':>10}{'median ms':>11}{'min ms':>10}{'max ms':>10}")
    for stage, v in by_stage.items():
        print(f"  {stage:<26}{len(v):>6}{sum(v):>11.1f}{np.mean(v):>10.1f}{np.median(v):>11.1f}{min(v):>10.1f}{max(v):>10.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Directory with the BAGEL IRs (bagel_ov_config.json or optimum-intel export)")
    parser.add_argument("tokenizer_dir", help="Directory with openvino_tokenizer.xml / openvino_detokenizer.xml")
    parser.add_argument("device", help="Device to run the models on, e.g. NPU or CPU")
    parser.add_argument("task", choices=["vqa", "t2i", "edit"])
    parser.add_argument("prompt")
    parser.add_argument("--image", help="Input image (vqa, edit)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--size", type=int, default=512, help="Text-to-image side")
    parser.add_argument("--cfg-text", type=float, default=4.0)
    parser.add_argument("--cfg-img", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", default="bagel_out.png")
    parser.add_argument("--repeat", type=int, default=1, help="Run the task N times")
    parser.add_argument("--props", default="", help="Extra properties K=V;... applied on top of the per-model defaults")
    args = parser.parse_args()
    if args.task == "edit" and not args.image:
        parser.error("edit requires --image")

    props = dict(kv.split("=", 1) for kv in args.props.split(";") if kv)
    image = Tensor(np.array(Image.open(args.image).convert("RGB"))) if args.image else None

    config = openvino_genai.BagelImageGenerationConfig()
    config.num_inference_steps = args.steps
    config.size = args.size
    config.cfg_text_scale = args.cfg_text
    config.cfg_img_scale = args.cfg_img
    config.seed = args.seed

    pipe = openvino_genai.BagelPipeline(args.model_dir, args.device, args.tokenizer_dir, **props)

    for r in range(args.repeat):
        start = time.perf_counter()
        if args.task == "vqa":
            answer = pipe.generate_text(args.prompt, image, args.max_new_tokens)
            print(f"Q: {args.prompt}\nA: {answer}")
        else:
            result = pipe.generate_image(args.prompt, config) if args.task == "t2i" else pipe.edit_image(image, args.prompt, config)
            Image.fromarray(result.data[0]).save(args.out)
            print(f"saved {args.out} {result.shape[2]}x{result.shape[1]}")
        wall = time.perf_counter() - start

        perf = pipe.get_perf_metrics()
        variant_compile_s = sum(ms for name, ms in perf.compile_ms if not name.startswith("compile_model")) / 1e3
        print(f"\n== run {r}: wall {wall:.1f} s | variant compile {variant_compile_s:.1f} s | "
              f"inference {wall - perf.first_call_ms / 1e3:.1f} s")
        print_report(perf)
        print()
        pipe.reset_perf_metrics()


if __name__ == "__main__":
    main()
