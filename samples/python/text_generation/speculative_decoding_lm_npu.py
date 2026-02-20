#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai

def streamer(subword):
    print(subword, end="", flush=True)
    # Return flag corresponds whether generation should be stopped.
    return openvino_genai.StreamingStatus.RUNNING

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("draft_model_dir")
    parser.add_argument("prompt")
    args = parser.parse_args()

    # User can run main and draft model on different devices.
    # Please, set device for main model in `openvino_genai.LLMPipeline` constructor and in `openvino_genai.draft_model` for draft.
    # CPU, GPU and NPU can be used. For NPU, the preferred configuration is when both the main and draft models use NPU.
    main_device = "NPU"
    draft_device = "NPU"

    draft_model = openvino_genai.draft_model(args.draft_model_dir, draft_device)

    pipe = openvino_genai.LLMPipeline(args.model_dir, main_device, GENERATE_HINT="BEST_PERF", draft_model=draft_model)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 512
    # Speculative decoding generation parameters like `num_assistant_tokens` and `assistant_confidence_threshold` are mutually excluded.
    # Add parameter to enable speculative decoding to generate `num_assistant_tokens` candidates by draft_model per iteration.
    # NOTE: ContinuousBatching backend uses `num_assistant_tokens` as is. Stateful backend uses `num_assistant_tokens`'s copy as initial
    # value and adjusts it based on recent number of accepted tokens. If `num_assistant_tokens` is not set, it defaults to `5` for both
    # backends.
    config.num_assistant_tokens = 5
    # Add parameter to enable speculative decoding to generate candidates by draft_model while candidate probability is higher than
    # `assistant_confidence_threshold`.
    # NOTE: `assistant_confidence_threshold` is supported only by ContinuousBatching backend.
    # config.assistant_confidence_threshold = 0.4

    # Since the streamer is set, the results will be printed
    # every time a new token is generated and put into the streamer queue.
    res = pipe.generate([args.prompt], config, streamer)
    print()
    perf_metrics = res.perf_metrics
    print(f"Performance metrics:" )
    print(f"  Generate time: {perf_metrics.get_generate_duration().mean:.2f} ms" )
    print(f"  TTFT: {perf_metrics.get_ttft().mean:.2f} ms")
    print(f"  TPOT: {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms/token")
    print(f"  Throughput: {(1000.0 / perf_metrics.get_tpot().mean):.2f} token/s")
    print(f"  Num generated token: {perf_metrics.get_num_generated_tokens()} tokens")
    if res.extended_perf_metrics:
        print(f"  Num accepted token: {res.extended_perf_metrics.get_num_accepted_tokens()} tokens")
    print()


if "__main__" == __name__:
    main()
