#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
from video_utils import save_video


def print_perf_metrics(perf_metrics):
    print("\nPerformance metrics:")
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

    if len(adapters) % 2 != 0:
        parser.error(
            "Each LoRA adapter path must be followed by a numeric alpha value (got an odd number of extra arguments)."
        )

    # Multiple LoRA adapters applied simultaneously are supported, parse them all and corresponding alphas from cmd parameters:
    adapter_config = openvino_genai.AdapterConfig()
    for i in range(len(adapters) // 2):
        adapter_path = adapters[2 * i]
        alpha_str = adapters[2 * i + 1]
        try:
            alpha = float(alpha_str)
        except ValueError:
            parser.error(f"Invalid alpha value for LoRA adapter '{adapter_path}': '{alpha_str}' is not a number.")
        adapter_config.add(openvino_genai.Adapter(adapter_path), alpha)

    pipe = openvino_genai.Text2VideoPipeline(args.model_dir, "CPU", adapters=adapter_config)  # GPU can be used as well

    def callback(step, num_steps, latent):
        print(f"Generation step {step + 1} / {num_steps}")
        return False

    frame_rate = 25

    generate_args = dict(
        negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
        height=480,
        num_inference_steps=25,
        callback=callback,
        guidance_scale=3,
    )

    print("Generating video with LoRA adapters applied, resulting video will be in lora_video.avi")
    output = pipe.generate(args.prompt, **generate_args)
    save_video("lora_video.avi", output.video, frame_rate)
    print_perf_metrics(output.perf_metrics)

    print("Generating video without LoRA adapters applied, resulting video will be in baseline_video.avi")
    output = pipe.generate(
        args.prompt,
        # passing adapters in generate overrides adapters set in the constructor; openvino_genai.AdapterConfig() means no adapters
        adapters=openvino_genai.AdapterConfig(),
        **generate_args,
    )
    save_video("baseline_video.avi", output.video, frame_rate)
    print_perf_metrics(output.perf_metrics)


if __name__ == "__main__":
    main()
