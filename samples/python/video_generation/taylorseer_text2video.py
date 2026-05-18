#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import time
import openvino_genai
from video_utils import save_video


def main():
    parser = argparse.ArgumentParser(description="Text-to-video generation with TaylorSeer caching optimization")
    parser.add_argument("model_dir", help="Path to the converted OpenVINO model directory")
    parser.add_argument("prompt", help="Text prompt for video generation")
    args = parser.parse_args()

    device = "CPU"  # GPU can be used as well
    pipe = openvino_genai.Text2VideoPipeline(args.model_dir, device)
    frame_rate = pipe.get_generation_config().frame_rate

    # TaylorSeer configuration
    cache_interval = 3
    disable_before = 6
    disable_after = -2
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    num_inference_steps = 25

    def callback(step, num_steps, latent):
        print(f"Step {step + 1}/{num_steps}")
        return False

    generate_kwargs = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "callback": callback,
    }

    # Generate baseline for comparison
    print(f"\nGenerating baseline video without caching...")
    baseline_config = pipe.get_generation_config()
    baseline_config.taylorseer_config = None  # explicitly disable caching
    pipe.set_generation_config(baseline_config)

    start_time = time.time()
    baseline_output = pipe.generate(args.prompt, **generate_kwargs)
    baseline_time = time.time() - start_time

    print(f"Baseline generation completed in {baseline_time:.2f}s")

    baseline_filename = "taylorseer_baseline.avi"
    save_video(baseline_filename, baseline_output.video, frame_rate)
    print(f"Baseline video saved to {baseline_filename}")

    # Configure TaylorSeer caching
    print(f"\nGenerating video with TaylorSeer caching...")

    taylorseer_config = openvino_genai.TaylorSeerCacheConfig()
    taylorseer_config.cache_interval = cache_interval
    taylorseer_config.disable_cache_before_step = disable_before
    taylorseer_config.disable_cache_after_step = disable_after
    print(taylorseer_config)

    start_time = time.time()
    output = pipe.generate(args.prompt, taylorseer_config=taylorseer_config, **generate_kwargs)
    taylorseer_time = time.time() - start_time
    print(f"TaylorSeer generation completed in {taylorseer_time:.2f}s")

    video_filename = "taylorseer.avi"
    save_video(video_filename, output.video, frame_rate)
    print(f"Video saved to {video_filename}")

    # Performance comparison
    speedup = baseline_time / taylorseer_time if taylorseer_time > 0 else 0.0
    time_saved = baseline_time - taylorseer_time if baseline_time > 0 else 0.0
    percentage = (baseline_time - taylorseer_time) / baseline_time * 100 if baseline_time > 0 else 0.0

    print(f"\nPerformance Comparison:")
    print(f"  Baseline time: {baseline_time:.2f}s")
    print(f"  TaylorSeer time: {taylorseer_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time saved: {time_saved:.2f}s ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
