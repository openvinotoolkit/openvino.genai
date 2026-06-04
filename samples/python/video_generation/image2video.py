#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import numpy as np
import openvino as ov
import openvino_genai
from PIL import Image
from video_utils import save_video


def read_image(path: str) -> ov.Tensor:
    pic = Image.open(path).convert("RGB")
    return ov.Tensor(np.array(pic))  # [H, W, 3] uint8; pipeline handles resizing


def main():
    parser = argparse.ArgumentParser(description="Generate video from an image and text prompt using OpenVINO GenAI")
    parser.add_argument("model_dir", help="Path to the model directory (must contain vae_encoder/)")
    parser.add_argument("image", help="Path to the conditioning image")
    parser.add_argument("prompt", help="Text prompt to guide generation")
    args = parser.parse_args()

    pipe = openvino_genai.Image2VideoPipeline(args.model_dir, "CPU")  # GPU can be used as well

    frame_rate = 25

    def callback(step, num_steps, latent):
        print(f"Generation step {step + 1} / {num_steps}")
        return False

    output = pipe.generate(
        read_image(args.image),
        args.prompt,
        negative_prompt="static, motionless, frozen, still photograph, no movement, low quality, blurry, distorted",
        height=480,
        width=704,
        num_frames=161,
        num_inference_steps=50,
        num_videos_per_prompt=1,
        callback=callback,
        frame_rate=frame_rate,
        guidance_scale=4.0,
        generator=openvino_genai.CppStdGenerator(42),
    )

    save_video("genai_video.avi", output.video, frame_rate)

    print(f"\nPerformance metrics:")
    print(f"  Load time: {output.perf_metrics.get_load_time():.2f} ms")
    print(f"  Generate duration: {output.perf_metrics.get_generate_duration():.2f} ms")
    print(f"  Transformer duration: {output.perf_metrics.get_transformer_infer_duration().mean:.2f} ms")
    print(f"  VAE decoder duration: {output.perf_metrics.get_vae_decoder_infer_duration():.2f} ms")


if __name__ == "__main__":
    main()
