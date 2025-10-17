#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai


def streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped.
    return openvino_genai.StreamingStatus.RUNNING

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='Path to the model directory')
    parser.add_argument('device', nargs='?', default='CPU', help='Device to run the model on (default: CPU)')
    args = parser.parse_args()

    device = args.device
    pipe = openvino_genai.LLMPipeline(args.model_dir, device, {"ATTENTION_BACKEND" : "SDPA"})

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 150
    config.do_sample = False
    config.apply_chat_template = True

    # pipe.start_chat()
    while True:
        try:
            prompt = input('question:\n')
        except EOFError:
            break
        res = pipe.generate([prompt], config, streamer)
        print()
        perf_metrics = res.perf_metrics
        print(f"Stateful pipeline" )
        print(f"  Generate time: {perf_metrics.get_generate_duration().mean:.2f} ms" )
        print(f"  TTFT: {perf_metrics.get_ttft().mean:.2f} ms")
        print(f"  TPOT: {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms/token")
        print(f"  Throughput: {(1000.0 / perf_metrics.get_tpot().mean):.2f}")
        print(f"  Num generated token: {perf_metrics.get_num_generated_tokens()} tokens")
        print(f"  Total iteration number: {len(perf_metrics.raw_metrics.m_durations)}")
        print()
        print('\n----------')
    # pipe.finish_chat()


if '__main__' == __name__:
    main()
