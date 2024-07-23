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
    parser.add_argument("-n", "--num_iter", type=int, default=5, help="Number of iterations")
    parser.add_argument("-mt", "--max_new_tokens", type=int, default=20, help="Maximal number of new tokens")
    parser.add_argument("-d", "--device", type=str, default="CPU", help="Device")
    
    args = parser.parse_args()

    # Perf metrics is stored in DecodedResults. 
    # In order to get DecodedResults instead of a string input should be a list.
    
    model_path = args.model
    device = args.device
    num_warmup = args.num_warmup
    num_iter = args.num_iter
    
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 20
    # config.num_beam_groups = 3
    # config.num_beams = 15

    pipe = ov_genai.LLMPipeline(model_path, device)
    
    import pandas as pd
    metrics_df = pd.DataFrame(columns=['batch_size', 'throughput', 'ttft', 'tpot', 'std_throughput', 'std_ttft', 'std_tpot'])

    batch_sizes = [1, 2, 4, 16, 32, 64, 256]
    for batch_size in batch_sizes:
        prompt = [args.prompt] * batch_size
        for _ in range(num_warmup):
            pipe.generate(prompt, config)
        
        res = pipe.generate(prompt, config)
        metrics = res.metrics
        for _ in range(num_iter - 1):
            res = pipe.generate(prompt, config)
            metrics += res.metrics
        # pdb.set_trace()
        metrics_df = metrics_df._append({
            'batch_size': batch_size,
            'throughput': metrics.mean_throughput,
            'ttft': metrics.mean_ttft,
            'tpot': metrics.mean_tpot,
            'std_throughput': metrics.std_throughput,
            'std_ttft': metrics.std_ttft,
            'std_tpot': metrics.std_tpot,
        }, ignore_index=True)

    metrics_df.to_csv('metrics.csv', index=False)

if __name__ == "__main__":
    main()
