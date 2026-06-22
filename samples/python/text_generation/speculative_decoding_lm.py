#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai

def streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped.
    return openvino_genai.StreamingStatus.RUNNING

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('draft_model_dir')
    parser.add_argument('prompt')
    args = parser.parse_args()

    # User can run main and draft model on different devices.
    # Please, set device for main model in `openvino_genai.LLMPipeline` constructor and in `openvino_genai.draft_model` for draft.
    # CPU, GPU and NPU can be used. For NPU, the preferred configuration is when both the main and draft models use NPU.
    main_device = 'CPU'
    draft_device = 'CPU'

    # Select speculative decoding approach:
    #   'fast_draft' - sequential candidate generation with a smaller draft model
    #   'eagle3'     - Eagle3 tree-based candidate generation
    speculative_mode = "fast_draft"

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    draft_properties = {}

    if speculative_mode == "fast_draft":
        # Speculative decoding generation parameters like `num_assistant_tokens` and `assistant_confidence_threshold`
        # are mutually exclusive.
        # `num_assistant_tokens` controls how many candidates the draft model generates per iteration.
        # NOTE: ContinuousBatching backend uses `num_assistant_tokens` as is. Stateful backend uses it as initial
        # value and adjusts based on recent number of accepted tokens. Defaults to `5` for both backends.
        config.num_assistant_tokens = 4
        # `assistant_confidence_threshold` generates candidates while probability exceeds the threshold.
        # NOTE: supported only by ContinuousBatching backend.
        # config.assistant_confidence_threshold = 0.4

    elif speculative_mode == "eagle3":
        # Eagle3 tree-based speculative decoding parameters:
        # `branching_factor` - number of top-k candidates selected per tree node and kept globally per tree layer.
        # `tree_depth` - lookahead depth of the candidate tree; the draft model runs `tree_depth` iterations.
        # `num_assistant_tokens` - number of candidate (non-root) tokens submitted to the target model for
        #   verification; total tree nodes = `num_assistant_tokens + 1` (including root).
        # NOTE: The total draft tokens produced by the tree is:
        #   total_draft_tokens = branching_factor^2 * (tree_depth - 1) + branching_factor
        # Constraint: total_draft_tokens >= num_assistant_tokens must hold.
        config.branching_factor = 4
        config.tree_depth = 2
        config.num_assistant_tokens = 7

        # NPU requires static tensor shapes at model compilation time, so compile-time upper bounds
        # for tree search dimensions must be specified. Runtime values must not exceed these limits.
        # On CPU/GPU these properties are ignored and can be omitted.
        if draft_device == "NPU":
            draft_properties = dict(
                MAX_TREE_DEPTH=config.tree_depth,
                MAX_BRANCHING_FACTOR=config.branching_factor,
                MAX_ASSISTANT_TOKENS=config.num_assistant_tokens,
            )

    draft_model = openvino_genai.draft_model(args.draft_model_dir, draft_device, **draft_properties)
    pipe = openvino_genai.LLMPipeline(args.model_dir, main_device, draft_model=draft_model)

    # Since the streamer is set, the results will be printed
    # every time a new token is generated and put into the streamer queue.
    res = pipe.generate([args.prompt], config, streamer)
    print()
    if (res.extended_perf_metrics):
        main_model_metrics = res.extended_perf_metrics.main_model_metrics
        print("MAIN MODEL")
        print(f"  Generate time: {main_model_metrics.get_generate_duration().mean:.2f} ms")
        print(f"  TTFT: {main_model_metrics.get_ttft().mean:.2f} ± {main_model_metrics.get_ttft().std:.2f} ms")
        print(f"  TTST: {main_model_metrics.get_ttst().mean:.2f} ± {main_model_metrics.get_ttst().std:.2f} ms/token")
        print(f"  TPOT: {main_model_metrics.get_tpot().mean:.2f} ± {main_model_metrics.get_tpot().std:.2f} ms/iteration")
        print(f"  AVG Latency: {main_model_metrics.get_latency().mean:.2f} ± {main_model_metrics.get_latency().std:.2f} ms/token")
        print(f"  Num generated token: {main_model_metrics.get_num_generated_tokens()} tokens")
        print(f"  Total iteration number: {len(main_model_metrics.raw_metrics.m_durations)}")
        print(f"  Num accepted token: {res.extended_perf_metrics.get_num_accepted_tokens()} tokens")

        draft_model_metrics = res.extended_perf_metrics.draft_model_metrics
        print("DRAFT MODEL")
        print(f"  Generate time: {draft_model_metrics.get_generate_duration().mean:.2f} ms")
        print(f"  TTFT: {draft_model_metrics.get_ttft().mean:.2f} ms")
        print(f"  TTST: {draft_model_metrics.get_ttst().mean:.2f} ms/token")
        print(f"  TPOT: {draft_model_metrics.get_tpot().mean:.2f} ± {draft_model_metrics.get_tpot().std:.2f} ms/token")
        print(f"  AVG Latency: {draft_model_metrics.get_latency().mean:.2f} ± {draft_model_metrics.get_latency().std:.2f} ms/iteration")
        print(f"  Num generated token: {draft_model_metrics.get_num_generated_tokens()} tokens")
        print(f"  Total iteration number: {len(draft_model_metrics.raw_metrics.m_durations)}")
        print()

if '__main__' == __name__:
    main()
