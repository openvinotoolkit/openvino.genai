#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
from video_utils import save_video


def print_perf_metrics(perf_metrics):
    print(f"\nPerformance metrics:")
    print(f"  Load time: {perf_metrics.get_load_time():.2f} ms")
    print(f"  Generate duration: {perf_metrics.get_generate_duration():.2f} ms")
    print(f"  Transformer duration: {perf_metrics.get_transformer_infer_duration().mean:.2f} ms")
    print(f"  VAE decoder duration: {perf_metrics.get_vae_decoder_infer_duration():.2f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="Generate video from text prompt using OpenVINO GenAI with LoRA adapters"
    )
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("prompt", help="Text prompt for video generation")
    args, adapters = parser.parse_known_args()

    # Multiple LoRA adapters applied simultaneously are supported, parse them all and corresponding alphas from cmd parameters:
    adapter_config = openvino_genai.AdapterConfig()
    for i in range(int(len(adapters) / 2)):
        adapter = openvino_genai.Adapter(adapters[2 * i])
        alpha = float(adapters[2 * i + 1])
        adapter_config.add(adapter, alpha)

    pipe = openvino_genai.Text2VideoPipeline(args.model_dir, "CPU", adapters=adapter_config)  # GPU can be used as well

    def callback(step, num_steps, latent):
        print(f"Generation step {step + 1} / {num_steps}")
        return False

    generate_args = dict(
        negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
        height=480,
        width=704,
        num_frames=161,
        num_inference_steps=25,
        num_videos_per_prompt=1,
        callback=callback,
        frame_rate=25,
        guidance_scale=3,
    )

    print("Generating video with LoRA adapters applied, resulting video will be in lora_video.avi")
    output = pipe.generate(args.prompt, **generate_args)
    save_video("lora_video.avi", output.video, 25)
    print_perf_metrics(output.perf_metrics)

    print("Generating video without LoRA adapters applied, resulting video will be in baseline_video.avi")
    output = pipe.generate(
        args.prompt,
        # passing adapters in generate overrides adapters set in the constructor; openvino_genai.AdapterConfig() means no adapters
        adapters=openvino_genai.AdapterConfig(),
        **generate_args,
    )
    save_video("baseline_video.avi", output.video, 25)
    print_perf_metrics(output.perf_metrics)


if __name__ == "__main__":
    main()
