import sys
import argparse
import openvino_genai as ov_genai
from openvino import get_version
import hashlib
import os

def main():
    parser = argparse.ArgumentParser(description="Help command")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to model and tokenizers base directory")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="Prompt")
    parser.add_argument("-pd", "--prompt_dir", type=str, required=True, help="Directory containing prompt files")
    parser.add_argument("-nw", "--num_warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument("-n", "--num_iter", type=int, default=2, help="Number of iterations")
    parser.add_argument("-mt", "--max_new_tokens", type=int, default=128, help="Maximal number of new tokens")
    parser.add_argument("-d", "--device", type=str, default="CPU", help="Device")

    args = parser.parse_args()

    # List all files in the prompt directory
    prompt_files = [os.path.join(args.prompt_dir, f) for f in os.listdir(args.prompt_dir) if os.path.isfile(os.path.join(args.prompt_dir, f))]

    prompts = []
    for i, prompt_file in enumerate(prompt_files, start=1):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = [f.read()]
            if len(prompt) == 0:
                raise RuntimeError(f'Prompt {i} is empty!')
            prompts.append(prompt)

    print(f'openvino runtime version: {get_version()}')

    models_path = args.model
    device = args.device
    num_warmup = args.num_warmup
    num_iter = args.num_iter

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens

    ov_config = {}
    if device == 'NPU':
        ov_config["NPUW_LLM_ENABLE_PREFIX_CACHING"] = "YES"
        # ov_config["NPUW_LLM_ENABLE_PREFIX_CACHING"] = "NO"
        ov_config["MAX_PROMPT_LEN"] = 8192

    pipe = ov_genai.LLMPipeline(models_path, device, ov_config)

    for i, prompt in enumerate(prompts, start=1):
        print(f"\n")
        print(f"generate for prompt {i}")
        res = pipe.generate(prompt, config)
        print(res.texts[0])
        results_md5 = (hashlib.new("md5", res.texts[0].encode(), usedforsecurity=False).hexdigest())
        print(results_md5)

        perf_metrics = res.perf_metrics
        print(f"Output token size: {perf_metrics.get_num_generated_tokens()}")
        print(f"Load time: {perf_metrics.get_load_time():.2f} ms")
        print(f"Generate time: {perf_metrics.get_generate_duration().mean:.2f} ± {perf_metrics.get_generate_duration().std:.2f} ms")
        print(f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f} ± {perf_metrics.get_tokenization_duration().std:.2f} ms")
        print(f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f} ± {perf_metrics.get_detokenization_duration().std:.2f} ms")
        print(f"TTFT: {perf_metrics.get_ttft().mean:.2f} ± {perf_metrics.get_ttft().std:.2f} ms")
        print(f"TPOT: {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms")
        print(f"Throughput : {perf_metrics.get_throughput().mean:.2f} ± {perf_metrics.get_throughput().std:.2f} tokens/s")

if __name__ == "__main__":
    main()