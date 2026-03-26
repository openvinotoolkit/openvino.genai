#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import time

import openvino_genai
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Text-to-image generation with TaylorSeer caching optimization")
    parser.add_argument("model_dir", help="Path to the converted OpenVINO model directory")
    parser.add_argument("prompt", help="Text prompt for image generation")
    args = parser.parse_args()

    device = "CPU"  # GPU can be used as well
    pipe = openvino_genai.Text2ImagePipeline(args.model_dir, device)

    # TaylorSeer configuration
    cache_interval = 3
    disable_before = 6
    disable_after = -2
    num_inference_steps = 28

    def callback(step, num_steps, latent):
        print(f"Step {step + 1}/{num_steps}")
        return False

    generate_kwargs = {
        "width": 512,
        "height": 512,
        "num_inference_steps": num_inference_steps,
        "rng_seed": 42,
        "num_images_per_prompt": 1,
        "callback": callback,
    }

    # Generate baseline for comparison
    print("\nGenerating baseline image without caching...")
    start_time = time.time()
    baseline_tensor = pipe.generate(args.prompt, **generate_kwargs)
    baseline_time = time.time() - start_time

    print(f"Baseline generation completed in {baseline_time:.2f}s")

    baseline_filename = "taylorseer_baseline.bmp"
    baseline_image = Image.fromarray(baseline_tensor.data[0])
    baseline_image.save(baseline_filename)
    print(f"Baseline image saved to {baseline_filename}")

    # Configure TaylorSeer caching
    print("\nGenerating image with TaylorSeer caching...")

    taylorseer_config = openvino_genai.TaylorSeerCacheConfig()
    taylorseer_config.cache_interval = cache_interval
    taylorseer_config.disable_cache_before_step = disable_before
    taylorseer_config.disable_cache_after_step = disable_after
    print(taylorseer_config)
    generation_config = pipe.get_generation_config()
    generation_config.taylorseer_config = taylorseer_config
    pipe.set_generation_config(generation_config)

    start_time = time.time()
    image_tensor = pipe.generate(args.prompt, **generate_kwargs)
    taylorseer_time = time.time() - start_time
    print(f"TaylorSeer generation completed in {taylorseer_time:.2f}s")

    image_filename = "taylorseer.bmp"
    image = Image.fromarray(image_tensor.data[0])
    image.save(image_filename)
    print(f"Image saved to {image_filename}")

    # Performance comparison
    speedup = baseline_time / taylorseer_time if taylorseer_time > 0 else 0.0
    time_saved = baseline_time - taylorseer_time if baseline_time > 0 else 0.0
    percentage = (baseline_time - taylorseer_time) / baseline_time * 100 if baseline_time > 0 else 0.0

    print("\nPerformance Comparison:")
    print(f"  Baseline time: {baseline_time:.2f}s")
    print(f"  TaylorSeer time: {taylorseer_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time saved: {time_saved:.2f}s ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
