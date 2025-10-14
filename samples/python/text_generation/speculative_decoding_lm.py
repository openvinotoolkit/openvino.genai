#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
import queue

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
    # Please, set device for main model in `openvino_genai.LLMPipeline` constructor and in openvino_genai.draft_model` for draft.
    main_device = 'NPU'  # GPU can be used as well
    draft_device = 'NPU'

    draft_model = openvino_genai.draft_model(args.draft_model_dir, draft_device)

    pipe = openvino_genai.LLMPipeline(args.model_dir, main_device, config={"GENERATE_HINT" : "BEST_PERF"}, draft_model=draft_model)
    
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 150
    # Speculative decoding generation parameters like `num_assistant_tokens` and `assistant_confidence_threshold` are mutually excluded
    # add parameter to enable speculative decoding to generate `num_assistant_tokens` candidates by draft_model per iteration
    config.num_assistant_tokens = 5
    # add parameter to enable speculative decoding to generate candidates by draft_model while candidate probability is higher than `assistant_confidence_threshold`
    # config.assistant_confidence_threshold = 0.4

    # Since the streamer is set, the results will be printed 
    # every time a new token is generated and put into the streamer queue.
    res = pipe.generate([args.prompt], config, streamer)
    print()
    if (res.extended_perf_metrics):
        main_model_metrics = res.extended_perf_metrics.main_model_metrics
        print(f"MAIN MODEL")
        print(f"  Generate time: {main_model_metrics.get_generate_duration().mean:.2f} ms" )
        print(f"  TTFT: {main_model_metrics.get_ttft().mean:.2f} ± {main_model_metrics.get_ttft().std:.2f} ms" )
        print(f"  TTST: {main_model_metrics.get_ttst().mean:.2f} ± {main_model_metrics.get_ttst().std:.2f} ms/token")
        print(f"  TPOT: {main_model_metrics.get_tpot().mean:.2f} ± {main_model_metrics.get_tpot().std:.2f} ms/iteration")
        print(f"  AVG Latency: {main_model_metrics.get_latency().mean:.2f} ± {main_model_metrics.get_latency().std:.2f} ms/token")
        print(f"  Num generated token: {main_model_metrics.get_num_generated_tokens()} tokens")
        print(f"  Total iteration number: {len(main_model_metrics.raw_metrics.m_durations)}")
        print(f"  Num accepted token: {res.extended_perf_metrics.get_num_accepted_tokens()} tokens")

        draft_model_metrics = res.extended_perf_metrics.draft_model_metrics
        print(f"DRAFT MODEL" )
        print(f"  Generate time: {draft_model_metrics.get_generate_duration().mean:.2f} ms" )
        print(f"  TTFT: {draft_model_metrics.get_ttft().mean:.2f} ms")
        print(f"  TTST: {draft_model_metrics.get_ttst().mean:.2f} ms/token")
        print(f"  TPOT: {draft_model_metrics.get_tpot().mean:.2f} ± {draft_model_metrics.get_tpot().std:.2f} ms/token")
        print(f"  AVG Latency: {draft_model_metrics.get_latency().mean:.2f} ± {draft_model_metrics.get_latency().std:.2f} ms/iteration")
        print(f"  Num generated token: {draft_model_metrics.get_num_generated_tokens()} tokens")
        print(f"  Total iteration number: {len(draft_model_metrics.raw_metrics.m_durations)}")
        print()
    perf_metrics = res.perf_metrics
    print(f"Whole pipeline" )
    print(f"  Generate time: {perf_metrics.get_generate_duration().mean:.2f} ms" )
    print(f"  TTFT: {perf_metrics.get_ttft().mean:.2f} ms")
    print(f"  TPOT: {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms/token")
    print(f"  Throughput: {(1000.0 / perf_metrics.get_tpot().mean):.2f}")
    print(f"  Num generated token: {perf_metrics.get_num_generated_tokens()} tokens")
    print(f"  Total iteration number: {len(perf_metrics.raw_metrics.m_durations)}")
    print()

if '__main__' == __name__:
    main()
