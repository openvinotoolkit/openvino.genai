# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai as ov_genai
import pdb

def main():
    parser = argparse.ArgumentParser(description="Help command")
    parser.add_argument("-m", "--model", type=str, help="Path to model and tokenizers base directory")
    parser.add_argument("-p", "--prompt", type=str, default="The Sky is blue because", help="Prompt")
    parser.add_argument("-nw", "--num_warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument("-n", "--num_iter", type=int, default=3, help="Number of iterations")
    parser.add_argument("-mt", "--max_new_tokens", type=int, default=20, help="Maximal number of new tokens")
    parser.add_argument("-d", "--device", type=str, default="CPU", help="Device")
    
    args = parser.parse_args()

    prompt = [args.prompt]
    model_path = args.model
    device = args.device
    num_warmup = args.num_warmup
    num_iter = args.num_iter
    

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.num_new_tokens

    pipe = ov_genai.LLMPipeline(model_path, device)
    
    for _ in range(num_warmup):
        pipe.generate(prompt, config)
    
    res = pipe.generate(prompt, config)
    metrics = res.metrics
    for _ in range(num_iter - 1):
        # pdb.set_trace()
        res = pipe.generate(prompt, config)
        metrics += res.metrics

    print(f"Load time: {metrics.load_time} ms")
    print(f"Generate time: {metrics.mean_generate_duration:.2f} ± {metrics.std_generate_duration:.2f} ms")
    print(f"Tokenization time: {metrics.mean_tokenization_duration:.2f} ± {metrics.std_tokenization_duration:.2f} ms")
    print(f"Detokenization time: {metrics.mean_detokenization_duration:.2f} ± {metrics.std_detokenization_duration:.2f} ms")
    print(f"TTFT: {metrics.mean_ttft:.2f} ± {metrics.std_ttft:.2f} ms")
    print(f"TPOT: {metrics.mean_tpot:.2f} ± {metrics.std_tpot:.2f} ms")
    print(f"Throughput tokens/s: {metrics.mean_throughput:.2f} ± {metrics.std_throughput:.2f}")

if __name__ == "__main__":
    main()
