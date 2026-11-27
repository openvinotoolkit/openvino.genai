#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import statistics
import time
from collections import defaultdict

import numpy as np
import openvino_genai
from openvino import Tensor
from PIL import Image

# as the original OpenVINO BAGEL world model demo
FRAME_LABELS = ["Frame 1 (t-2):", "Frame 2 (t-1):", "Frame 3 (t, current):"]


def run_task(pipe, args, images, config):
    if args.task == "vqa":
        return pipe.generate_text(args.prompt, images)
    if args.task == "frames":
        # the frames that follow the input frames, decoded together at the end
        return pipe.generate_frames(images, args.prompt, args.num_frames, config)
    if args.task == "t2i":
        return pipe.generate_image(args.prompt, config)
    return pipe.edit_image(images, args.prompt, config)


def print_benchmark(compile_ms, perf, latencies):
    """compile time, latency over the timed iterations and the stages of one iteration"""
    iterations = len(latencies)
    print(f"\ncompile: {sum(ms for _, ms in compile_ms) / 1e3:.1f} s")
    for name, ms in compile_ms:
        print(f"  {name:<48}{ms:>10.1f} ms")
    print(
        f"latency over {iterations} iterations: mean {statistics.mean(latencies):.3f} s, "
        f"median {statistics.median(latencies):.3f} s, min {min(latencies):.3f} s, max {max(latencies):.3f} s"
    )
    stages = defaultdict(list)
    for stage, ms in perf.infer_ms:
        stages[stage].append(ms)
    # a denoising step spans language model calls, so steps are reported apart from the stage shares
    steps = {stage: ms for stage, ms in stages.items() if stage.startswith("step.")}
    calls = {stage: ms for stage, ms in stages.items() if not stage.startswith("step.")}
    total = sum(sum(ms) for ms in calls.values())
    print(f"  {'stage':<28}{'calls/iter':>11}{'ms/call':>10}{'ms/iter':>11}{'share':>8}")
    for stage, ms in sorted(calls.items(), key=lambda item: -sum(item[1])):
        print(
            f"  {stage:<28}{len(ms) / iterations:>11.1f}{statistics.mean(ms):>10.1f}"
            f"{sum(ms) / iterations:>11.1f}{sum(ms) / total:>8.1%}"
        )
    for stage, ms in steps.items():
        print(f"  {stage:<28}{len(ms) / iterations:>11.1f}{statistics.mean(ms):>10.1f}{sum(ms) / iterations:>11.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("device", help="Device to run the model on (e.g. CPU, GPU, NPU)")
    parser.add_argument("task", choices=["vqa", "t2i", "edit", "frames"])
    parser.add_argument("prompt")
    parser.add_argument("images", nargs="*", help="Input images (vqa, at least one for edit and frames)")
    parser.add_argument("--steps", type=int, default=50, help="Denoising steps (t2i, edit, frames)")
    parser.add_argument("--cfg-text", type=float, default=4.0, help="Text guidance scale, 1 disables guidance")
    parser.add_argument("--cfg-img", type=float, default=2.0, help="Image guidance scale (edit, frames), 1 disables it")
    parser.add_argument(
        "--cfg-renorm",
        choices=["global", "channel", "text_channel"],
        help="Guidance renormalization; defaults to text_channel for edit and frames (as BAGEL editing), else global",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=3, help="Frames to generate (frames)")
    parser.add_argument(
        "--frame-labels",
        nargs="*",
        help="Text before each input frame (frames), one per frame; defaults to 'Frame 1 (t-2):' .. for 3 frames",
    )
    parser.add_argument("--image-size", type=int, help="Resize the input images to SIZE x SIZE")
    parser.add_argument("--props", default="", help="Device properties K=V;K=V, e.g. NPUW_BAGEL_MAX_LM_VARIANTS=2")
    parser.add_argument("--config-file", help="JSON file of compile properties, for all models or per model")
    parser.add_argument("--warmup", type=int, default=0, help="Untimed runs before the timed ones")
    parser.add_argument("--iterations", type=int, default=0, help="Timed runs; prints a benchmark report if set")
    args = parser.parse_args()
    if args.task in ("edit", "frames") and not args.images:
        parser.error(f"{args.task} requires an image")

    props = dict(kv.split("=", 1) for kv in args.props.split(";") if kv)
    if args.config_file:
        props["BAGEL_CONFIG_FILE"] = args.config_file
    pipe = openvino_genai.BagelPipeline(args.model_dir, args.device, **props)
    pil_images = [Image.open(path).convert("RGB") for path in args.images]
    if args.image_size:
        pil_images = [image.resize((args.image_size, args.image_size), Image.BICUBIC) for image in pil_images]
    images = [Tensor(np.array(image)[None]) for image in pil_images]
    config = openvino_genai.BagelImageGenerationConfig()
    config.num_inference_steps = args.steps
    config.cfg_text_scale = args.cfg_text
    config.cfg_img_scale = args.cfg_img
    config.cfg_renorm_type = args.cfg_renorm or ("text_channel" if args.task in ("edit", "frames") else "global")
    config.seed = args.seed
    if args.task == "frames":
        if args.frame_labels is not None:
            config.frame_labels = args.frame_labels
        elif len(images) == len(FRAME_LABELS):
            config.frame_labels = FRAME_LABELS

    for _ in range(args.warmup):
        run_task(pipe, args, images, config)
    compile_ms = pipe.get_perf_metrics().compile_ms
    pipe.reset_perf_metrics()
    latencies = []
    for _ in range(max(args.iterations, 1)):
        start = time.perf_counter()
        result = run_task(pipe, args, images, config)
        latencies.append(time.perf_counter() - start)

    if args.task == "vqa":
        print(result)
    elif args.task == "frames":
        for i, frame in enumerate(result.data):
            Image.fromarray(frame).save(f"next_frame_{i}.bmp")
    else:
        Image.fromarray(result.data[0]).save("image.bmp")
    if args.iterations:
        if not args.warmup:
            # without warmup, the first-use compilation of the NPU variants is part of the timed run
            compile_ms = compile_ms + pipe.get_perf_metrics().compile_ms
        print_benchmark(compile_ms, pipe.get_perf_metrics(), latencies)


if __name__ == "__main__":
    main()
