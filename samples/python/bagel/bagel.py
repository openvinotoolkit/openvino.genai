#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino_genai
from openvino import Tensor
from PIL import Image


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=3, help="Frames to generate (frames)")
    args = parser.parse_args()
    if args.task in ("edit", "frames") and not args.images:
        parser.error(f"{args.task} requires an image")

    pipe = openvino_genai.BagelPipeline(args.model_dir, args.device)
    images = [Tensor(np.array(Image.open(path).convert("RGB"))[None]) for path in args.images]
    config = openvino_genai.BagelImageGenerationConfig()
    config.num_inference_steps = args.steps
    config.cfg_text_scale = args.cfg_text
    config.cfg_img_scale = args.cfg_img
    config.seed = args.seed

    if args.task == "vqa":
        print(pipe.generate_text(args.prompt, images))
        return
    if args.task == "frames":
        # the frames that follow the input frames, decoded together at the end
        for i, frame in enumerate(pipe.generate_frames(images, args.prompt, args.num_frames, config).data):
            Image.fromarray(frame).save(f"next_frame_{i}.bmp")
        return
    if args.task == "t2i":
        result = pipe.generate_image(args.prompt, config)
    else:
        result = pipe.edit_image(images, args.prompt, config)
    Image.fromarray(result.data[0]).save("image.bmp")


if __name__ == "__main__":
    main()
