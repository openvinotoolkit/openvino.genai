#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import time

import openvino_genai


def streamer(subword):
    print(subword, end="", flush=True)
    # Return flag corresponds whether generation should be stopped.
    return openvino_genai.StreamingStatus.RUNNING


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument(
        "device",
        nargs="?",
        default="CPU",
        help="Device to run the model on (default: CPU)",
    )
    parser.add_argument(
        "-mt",
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate (default: 100)",
    )
    parser.add_argument(
        "-t",
        "--num_threads",
        type=int,
        default=None,
        help="Number of threads to use for inference (CPU only, ignored for GPU/NPU)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display performance stats after each generation",
    )
    args = parser.parse_args()

    device = args.device

    # Configure properties for the pipeline
    properties = {}
    # Only apply thread count for CPU devices
    if device.upper() == "CPU" and args.num_threads is not None:
        properties["INFERENCE_NUM_THREADS"] = args.num_threads

    pipe = openvino_genai.LLMPipeline(args.model_dir, device, **properties)

    config = openvino_genai.GenerationConfig()

    config.max_new_tokens = args.max_new_tokens

    pipe.start_chat()
    while True:
        try:
            prompt = input("question:\n")
        except EOFError:
            break

        # Track generation time if verbose mode is enabled
        if args.verbose:
            start_time = time.time()
            # Pass prompt as a list to get DecodedResults with performance metrics
            result = pipe.generate([prompt], config, streamer)
            end_time = time.time()

            generation_time = end_time - start_time

            # Get accurate token count and performance metrics
            perf_metrics = result.perf_metrics
            tokens_generated = perf_metrics.get_num_generated_tokens()
            ttft = perf_metrics.get_ttft().mean  # Time to first token
            tpot = perf_metrics.get_tpot().mean  # Time per output token
            throughput = perf_metrics.get_throughput().mean  # Tokens per second

            print(f"\n----------")
            print(f"Performance Stats:")
            print(f"  Generation time: {generation_time:.2f} seconds")
            print(f"  Tokens generated: {tokens_generated}")
            print(f"  Time to first token (TTFT): {ttft:.2f} ms")
            print(f"  Time per output token (TPOT): {tpot:.2f} ms")
            print(f"  Throughput: {throughput:.2f} tokens/s")
            print(f"----------")
        else:
            pipe.generate(prompt, config, streamer)
            print("\n----------")

    pipe.finish_chat()


if "__main__" == __name__:
    main()
