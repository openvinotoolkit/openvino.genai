# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai as ov_genai

def main():
    parser = argparse.ArgumentParser(description="Help command")
    parser.add_argument("-m", "--model", type=str, help="Path to model and tokenizers base directory")
    parser.add_argument("-p", "--prompt", type=str, default="The Sky is blue because", help="Prompt")
    parser.add_argument("-nw", "--num_warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument("-n", "--num_iter", type=int, default=2, help="Number of iterations")
    parser.add_argument("-mt", "--max_new_tokens", type=int, default=20, help="Maximal number of new tokens")
    parser.add_argument("-d", "--device", type=str, default="CPU", help="Device")
    
    args = parser.parse_args()

    # Perf metrics is stored in DecodedResults. 
    # In order to get DecodedResults instead of a string input should be a list.
    prompt = [args.prompt]
    model_path = args.model
    device = args.device
    num_warmup = args.num_warmup
    num_iter = args.num_iter
    
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens

    pipe = ov_genai.LLMPipeline(model_path, device)
    
    for _ in range(num_warmup):
        pipe.generate(prompt, config)
    
    res = pipe.generate(prompt, config)
    perf_metrics = res.perf_metrics
    for _ in range(num_iter - 1):
        res = pipe.generate(prompt, config)
        perf_metrics += res.perf_metrics
    
    print(f"Load time: {perf_metrics.load_time:.2f} ms")
    print(f"Generate time: {perf_metrics.generate_duration.mean:.2f} ± {perf_metrics.generate_duration.std:.2f} ms")
    print(f"Tokenization time: {perf_metrics.tokenization_duration.mean:.2f} ± {perf_metrics.tokenization_duration.std:.2f} ms")
    print(f"Detokenization time: {perf_metrics.detokenization_duration.mean:.2f} ± {perf_metrics.detokenization_duration.std:.2f} ms")
    print(f"TTFT: {perf_metrics.ttft.mean:.2f} ± {perf_metrics.ttft.std:.2f} ms")
    print(f"TPOT: {perf_metrics.tpot.mean:.2f} ± {perf_metrics.tpot.std:.2f} ms")
    print(f"Throughput : {perf_metrics.throughput.mean:.2f} ± {perf_metrics.throughput.std:.2f} tokens/s")

if __name__ == "__main__":
    main()
