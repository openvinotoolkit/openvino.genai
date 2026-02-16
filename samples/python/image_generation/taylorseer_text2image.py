#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import time

import openvino_genai
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Text-to-image generation with TaylorSeer caching optimization")
    parser.add_argument('model_dir', help='Path to the converted OpenVINO model directory')
    parser.add_argument('prompt', help='Text prompt for image generation')
    parser.add_argument('--steps', type=int, default=28, help='Number of inference steps')

    ts_group = parser.add_argument_group('TaylorSeer Cache Configurations')
    ts_group.add_argument('--cache-interval', type=int, default=3, help='Cache interval')
    ts_group.add_argument('--disable-before', type=int, default=6,
                        help='Disable caching before this step for warmup')
    ts_group.add_argument('--disable-after', type=int, default=-2,
                        help='Disable caching after this step from end, -1 means last step')
    args = parser.parse_args()

    device = "CPU"  # GPU can be used as well
    pipe = openvino_genai.Text2ImagePipeline(args.model_dir, device)

    def callback(step, num_steps, latent):
        print(f"Step {step + 1}/{num_steps}")
        return False

    # Configure TaylorSeer caching
    taylorseer_config = openvino_genai.TaylorSeerCacheConfig(
        cache_interval=args.cache_interval,
        disable_cache_before_step=args.disable_before,
        disable_cache_after_step=args.disable_after
    )
    generation_config = pipe.get_generation_config()
    generation_config.taylorseer_config = taylorseer_config
    pipe.set_generation_config(generation_config)

    print(f"TaylorSeer Configuration:")
    print(f"  Cache interval: {args.cache_interval}")
    print(f"  Disable before step: {args.disable_before}")
    print(f"  Disable after step: {args.disable_after}")


    print(f"Generating image with TaylorSeer caching...")
    generate_kwargs = {
        'width': 512,
        'height': 512,
        'num_inference_steps': args.steps,
        'rng_seed': 42,
        'num_images_per_prompt': 1,
        'callback': callback,
    }

    start_time = time.time()
    image_tensor = pipe.generate(args.prompt, **generate_kwargs)
    taylorseer_time = time.time() - start_time
    print(f"TaylorSeer generation completed in {taylorseer_time:.2f}s")

    image_filename = "taylorseer.bmp"
    image = Image.fromarray(image_tensor.data[0])
    image.save(image_filename)
    print(f"Image saved to {image_filename}")

    print("\nGenerating baseline image without caching for comparison...")

    # Disable TaylorSeer by removing the config
    baseline_config = pipe.get_generation_config()
    baseline_config.taylorseer_config = None
    pipe.set_generation_config(baseline_config)

    start_time = time.time()
    baseline_tensor = pipe.generate(args.prompt, **generate_kwargs)
    baseline_time = time.time() - start_time

    print(f"Baseline generation completed in {baseline_time:.2f}s")

    baseline_filename = image_filename.replace('.bmp', '_baseline.bmp')
    baseline_image = Image.fromarray(baseline_tensor.data[0])
    baseline_image.save(baseline_filename)
    print(f"Baseline image saved to {baseline_filename}")

    # Performance comparison
    speedup = baseline_time / taylorseer_time
    time_saved = baseline_time - taylorseer_time
    print(f"\nPerformance Comparison:")
    print(f"  Baseline time: {baseline_time:.2f}s")
    print(f"  TaylorSeer time: {taylorseer_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time saved: {time_saved:.2f}s ({(time_saved/baseline_time*100):.1f}%)")


if __name__ == '__main__':
    main()
