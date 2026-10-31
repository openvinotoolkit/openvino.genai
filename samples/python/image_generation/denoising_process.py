#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "video_generation"))
from video_utils import save_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("prompt")
    args = parser.parse_args()

    rng_seed = 42
    num_inference_steps = 20
    image_width = 512
    image_height = 512
    fps = 5

    device = "CPU"  # GPU can be used as well
    pipe = openvino_genai.Text2ImagePipeline(args.model_dir, device)

    frames = np.zeros((1, num_inference_steps, image_height, image_width, 3), dtype=np.uint8)

    def callback(step, num_steps, latent):
        decoded = pipe.decode(latent)
        frames[0, step] = decoded.data[0]
        print(f"Step {step + 1}/{num_steps}")
        return False

    pipe.generate(
        args.prompt,
        width=image_width,
        height=image_height,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
        rng_seed=rng_seed,
        callback=callback,
    )

    save_video("denoising_process.avi", ov.Tensor(frames), fps)


if __name__ == "__main__":
    main()
