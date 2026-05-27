#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
from video_utils import save_video


def main():
    parser = argparse.ArgumentParser(description="Generate video from text prompt using OpenVINO GenAI")
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("prompt", help="Text prompt for video generation")
    args = parser.parse_args()

    pipe = openvino_genai.Text2VideoPipeline(args.model_dir, "CPU")  # GPU can be used as well

    frame_rate = 25

    def callback(step, num_steps, latent):
        print(f"Generation step {step + 1} / {num_steps}")
        return False

    output = pipe.generate(
        args.prompt,
        negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
        height=480,
        width=704,
        num_frames=161,
        num_inference_steps=25,
        num_videos_per_prompt=1,
        callback=callback,
        frame_rate=frame_rate,
        guidance_scale=3,
    )

    save_video("genai_video.avi", output.video, frame_rate)

    print(f"\nPerformance metrics:")
    print(f"  Load time: {output.perf_metrics.get_load_time():.2f} ms")
    print(f"  Generate duration: {output.perf_metrics.get_generate_duration():.2f} ms")
    print(f"  Transformer duration: {output.perf_metrics.get_transformer_infer_duration().mean:.2f} ms")
    print(f"  VAE decoder duration: {output.perf_metrics.get_vae_decoder_infer_duration():.2f} ms")


if __name__ == "__main__":
    main()
