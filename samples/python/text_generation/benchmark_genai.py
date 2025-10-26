# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys

import openvino_genai as ov_genai
from openvino import get_version


def main():
    parser = argparse.ArgumentParser(description="Help command")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to model and tokenizers base directory")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="Prompt")
    parser.add_argument("-pf", "--prompt_file", type=str, help="Read prompt from file")
    parser.add_argument("-nw", "--num_warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument("-n", "--num_iter", type=int, default=2, help="Number of iterations")
    parser.add_argument("-mt", "--max_new_tokens", type=int, default=20, help="Maximal number of new tokens")
    parser.add_argument("-d", "--device", type=str, default="CPU", help="Device")

    args = parser.parse_args()

    if args.prompt is not None and args.prompt_file is not None:
        raise RuntimeError("Cannot specify both --prompt and --prompt_file options simultaneously!")
    else:
        if args.prompt_file is not None:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompt = [f.read()]
        else:
            prompt = ["The Sky is blue because"] if args.prompt is None else [args.prompt]
    if len(prompt) == 0:
        raise RuntimeError("Prompt is empty!")

    print(f"openvino runtime version: {get_version()}, genai version: {ov_genai.__version__}")

    # Perf metrics is stored in DecodedResults.
    # In order to get DecodedResults instead of a string input should be a list.
    models_path = args.model
    device = args.device
    num_warmup = args.num_warmup
    num_iter = args.num_iter

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens
    config.apply_chat_template = False

    if device == "NPU":
        pipe = ov_genai.LLMPipeline(models_path, device)
    else:
        scheduler_config = ov_genai.SchedulerConfig()
        scheduler_config.enable_prefix_caching = False
        scheduler_config.max_num_batched_tokens = sys.maxsize
        pipe = ov_genai.LLMPipeline(models_path, device, scheduler_config=scheduler_config)

    input_data = pipe.get_tokenizer().encode(prompt)
    prompt_token_size = input_data.input_ids.get_shape()[1]
    print(f"Prompt token size: {prompt_token_size}")

    for _ in range(num_warmup):
        pipe.generate(prompt, config)

    res = pipe.generate(prompt, config)
    perf_metrics = res.perf_metrics
    for _ in range(num_iter - 1):
        res = pipe.generate(prompt, config)
        perf_metrics += res.perf_metrics

    print(f"Output token size: {res.perf_metrics.get_num_generated_tokens()}")
    print(f"Load time: {perf_metrics.get_load_time():.2f} ms")
    print(
        f"Generate time: {perf_metrics.get_generate_duration().mean:.2f} ± {perf_metrics.get_generate_duration().std:.2f} ms"
    )
    print(
        f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f} ± {perf_metrics.get_tokenization_duration().std:.2f} ms"
    )
    print(
        f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f} ± {perf_metrics.get_detokenization_duration().std:.2f} ms"
    )
    print(f"TTFT: {perf_metrics.get_ttft().mean:.2f} ± {perf_metrics.get_ttft().std:.2f} ms")
    print(f"TPOT: {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms")
    print(f"Throughput : {perf_metrics.get_throughput().mean:.2f} ± {perf_metrics.get_throughput().std:.2f} tokens/s")


if __name__ == "__main__":
    main()
