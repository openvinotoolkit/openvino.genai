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
    args = parser.parse_args()
    if args.task in ("edit", "frames") and not args.images:
        parser.error(f"{args.task} requires an image")

    pipe = openvino_genai.BagelPipeline(args.model_dir, args.device)
    images = [Tensor(np.array(Image.open(path).convert("RGB"))[None]) for path in args.images]

    if args.task == "vqa":
        print(pipe.generate_text(args.prompt, images))
        return
    if args.task == "frames":
        # the 3 frames that follow the input frames, decoded together at the end
        for i, frame in enumerate(pipe.generate_frames(images, args.prompt, 3).data):
            Image.fromarray(frame).save(f"next_frame_{i}.bmp")
        return
    result = pipe.generate_image(args.prompt) if args.task == "t2i" else pipe.edit_image(images, args.prompt)
    Image.fromarray(result.data[0]).save("image.bmp")


if __name__ == "__main__":
    main()
