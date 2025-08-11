
import gc
import os
import psutil
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from optimum.intel.openvino import OVModelForCausalLM
from openvino_genai import ContinuousBatchingPipeline, SchedulerConfig, GenerationResult, GenerationConfig, CacheEvictionConfig, AggregationMode
from openvino_tokenizers import convert_tokenizer
from openvino import serialize
from transformers import AutoTokenizer
import argparse

import time
import logging
from huggingface_hub.utils import HfHubHTTPError
from subprocess import CalledProcessError # nosec B404
from requests.exceptions import RequestException

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_request(func, retries=5):
    """
    Retries a function that makes a request up to a specified number of times.

    Parameters:
    func (callable): The function to be retried. It should be a callable that makes a request.
    retries (int): The number of retry attempts. Default is 5.

    Returns:
    Any: The return value of the function `func` if it succeeds.
    """
    network_error_patterns = [
        "ConnectionError",
        "Timeout",
        "Time-out",
        "ServiceUnavailable",
        "InternalServerError",
        "OSError",
        "HTTPError",
    ]

    for attempt in range(retries):
        try:
            return func()
        except (CalledProcessError, RequestException, HfHubHTTPError) as e:
            if isinstance(e, CalledProcessError):
                if e.stderr is not None and any(pattern in e.stderr for pattern in network_error_patterns):
                    logger.warning(f"CalledProcessError occurred: {e.stderr}")
                else:
                    raise
            if attempt < retries - 1:
                timeout = 2 ** attempt
                logger.info(f"Attempt {attempt + 1} failed. Retrying in {timeout} seconds.")
                time.sleep(timeout)
            else:
                raise

def load_prompts_dataset(file_name : str) -> dict[str, list[str]]:
    TESTS_ROOT = Path('tests/python_tests')
    file_path = TESTS_ROOT / 'data' / file_name
    with open(file_path, 'r') as f:
        return {"prompts": [s for s in f]}

def load_samsum_dataset(file_name : str) -> dict[str, list[str]]:
    import json
    retval = {"prompts": []}
    with open(file_name, 'r') as json_file:
        json_list = list(json_file)
        for json_str in json_list:
            result = json.loads(json_str)
            retval["prompts"].append(result["prompt"])
    return retval

def get_scheduler_config(num_kv_blocks: Optional[int]) -> SchedulerConfig:
    scheduler_config = SchedulerConfig()
    if num_kv_blocks is not None:
        scheduler_config.num_kv_blocks = num_kv_blocks
        scheduler_config.max_num_batched_tokens = 32 * num_kv_blocks
    scheduler_config.dynamic_split_fuse = True
    scheduler_config.max_num_seqs = 256
    scheduler_config.use_cache_eviction = False
    return scheduler_config

@dataclass
class ConvertedModel:
    model: OVModelForCausalLM
    tokenizer: AutoTokenizer
    models_path: Path


def get_converted_model(base_model_path: Path, model_id: str):
    model = retry_request(lambda: OVModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, load_in_8bit=False, compile=False, ov_config=get_default_llm_properties()))
    tokenizer = retry_request(lambda: AutoTokenizer.from_pretrained(model_id))
    models_path = base_model_path / model_id
    models_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(models_path)
    ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True, skip_special_tokens=True)
    serialize(ov_tokenizer, models_path / "openvino_tokenizer.xml")
    serialize(ov_detokenizer, models_path / "openvino_detokenizer.xml")
    converted_model = ConvertedModel(model, tokenizer, models_path)
    return converted_model


import openvino.properties.hint as hints
import openvino.properties as props
import openvino as ov

def get_default_llm_properties():
    return {
        hints.inference_precision : ov.Type.f32,
        hints.kv_cache_precision : ov.Type.f16,
    }

def run_and_write_metrics(model, prompt, generation_config, report_file):
    result: GenerationResult = model_cb_opt.generate([prompt], generation_config=[generation_config])

    pipeline_opt_metrics = model_cb_opt.get_metrics()
    rss_usage_gb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    result_length = len(result[0].m_generation_ids[0])
    print(f"avg_cache_usage:{pipeline_opt_metrics.avg_cache_usage:.2f}% max_cache_usage:{pipeline_opt_metrics.max_cache_usage:.2f}% rss_usage:{rss_usage_gb:.3f} GB")
    print(f"result length: {result_length}")
    print()

    if report_file is not None:
        with open(report_file, 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([generation_config.max_new_tokens - 1, result_length, pipeline_opt_metrics.avg_cache_usage, pipeline_opt_metrics.max_cache_usage, rss_usage_gb])
    return pipeline_opt_metrics.max_cache_usage


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eviction_on", action='store_true', help="Whether to apply cache eviction")
    parser.add_argument("--model", type=str, help="Model ID")
    parser.add_argument("--num_kv_blocks", type=int, help='Number of blocks to statically pre-allocate in cache.'
                                                          'If left unspecified, will allocate dynamically to accomodate the generation length.')
    parser.add_argument("--report", type=str, help="File name for CSV-formatted export of limit search data")
    parser.add_argument("--mode", type=str, nargs='?', choices=['gen_length', 'gen_throughput'], required=True)
    parser.add_argument("--data", type=str, help="Dataset jsonl file")
    parser.add_argument("--timeout", type=int, help="Maximum time allowed for a single round of generation in the `gen_length` mode", default=120)
    parser.add_argument("--device", type=str, help="Device for model inference", default="CPU")

    args = parser.parse_args()
    seqs_per_request = 1
    num_kv_blocks = args.num_kv_blocks

    scheduler_config_opt = get_scheduler_config(num_kv_blocks)
    if args.eviction_on:
        scheduler_config_opt.use_cache_eviction = True
        print("Eviction is ON")
    else:
        print("Eviction is OFF")

    base_model_path = Path("limit_checker_models")
    converted_model = get_converted_model(base_model_path, args.model)
    models_path = converted_model.models_path
    model_cb_opt = ContinuousBatchingPipeline(models_path, scheduler_config_opt, args.device, {}, get_default_llm_properties())

    tokenizer = converted_model.tokenizer
    if args.mode == "gen_length":
        data_dict = load_prompts_dataset('long_prompts.txt')
        prompt = data_dict["prompts"][0]

        generation_length = 1

        if args.report is not None:
            with open(args.report, 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['generation_length', 'result_length', 'avg_cache_usage_%', 'max_cache_usage_%', 'rss_usage_gb'])


        while True:
            gc.collect()
            generation_config = GenerationConfig()  # expecting default greedy sampling
            generation_config.num_return_sequences = 1
            generation_config.max_new_tokens = generation_length + 1
            generation_config.apply_chat_template = False
            generation_config.ignore_eos = True
            print(f"generation_length:{generation_length} ", sep='')

            start = time.time()
            max_cache_usage = run_and_write_metrics(model_cb_opt, prompt, generation_config, args.report)
            end = time.time()
            if (end - start) > args.timeout:
                print("Maximum generation time reached")
                break
            elif max_cache_usage == 100:
                print("Cache size exhausted")
                break

            generation_length *= 2

        del data_dict
    elif args.mode == "gen_throughput":
        dataset = load_samsum_dataset(args.data)
        prompt_throughput = 1
        prompt_left_bound = prompt_throughput
        prompt_right_bound = None
        is_right_bound = False

        while True:
            gc.collect()
            generation_config = GenerationConfig()  # expecting default greedy sampling
            generation_config.num_return_sequences = 1
            generation_config.apply_chat_template = False
            prompt_subset = dataset["prompts"][:prompt_throughput]
            print(f"prompt_throughput {prompt_throughput}")
            result: GenerationResult = model_cb_opt.generate(prompt_subset, generation_config=[generation_config] * len(prompt_subset))

            pipeline_opt_metrics = model_cb_opt.get_metrics()
            rss_usage_gb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
            print(f"avg_cache_usage:{pipeline_opt_metrics.avg_cache_usage:.2f}% max_cache_usage:{pipeline_opt_metrics.max_cache_usage:.2f}% rss_usage:{rss_usage_gb:.3f} GB")
            print()

            max_cache_usage = pipeline_opt_metrics.max_cache_usage

            if max_cache_usage == 100.0 and not is_right_bound:
                is_right_bound = True
                prompt_right_bound = prompt_throughput

            if not is_right_bound:
                prompt_left_bound = prompt_throughput
                prompt_throughput *= 2
            else:
                if max_cache_usage == 100.0:
                    prompt_right_bound = prompt_throughput
                elif max_cache_usage < 100.0:
                    prompt_left_bound = prompt_throughput
                prompt_throughput = (prompt_left_bound + prompt_right_bound) // 2

                if (prompt_right_bound - prompt_left_bound <= 1):
                    break


        print(f"Approximate highest throughput: {prompt_throughput} prompts")

