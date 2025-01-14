
from pathlib import Path
import sys
import psutil
from typing import Dict, List, Optional

import pathlib
from optimum.intel.openvino import OVModelForCausalLM

from openvino_genai import ContinuousBatchingPipeline, SchedulerConfig, GenerationResult, GenerationConfig, CacheEvictionConfig, AggregationMode

from openvino_tokenizers import convert_tokenizer
from openvino import serialize
from transformers import AutoTokenizer

def get_scheduler_config(num_kv_blocks: int) -> SchedulerConfig:
    scheduler_config = SchedulerConfig()
    scheduler_config.num_kv_blocks = num_kv_blocks
    scheduler_config.dynamic_split_fuse = True
    scheduler_config.max_num_batched_tokens = 256
    scheduler_config.max_num_seqs = 256
    scheduler_config.use_cache_eviction = False
    return scheduler_config

def get_default_properties():
    import openvino.properties.hint as hints
    import openvino as ov

    return {
        hints.inference_precision : ov.Type.f32,
        hints.kv_cache_precision : ov.Type.f16,
    }

def print_rss():
    process = psutil.Process()
    print(f"RSS usage: {process.memory_info().rss / 2 ** 30:.2f} GB")

def leak_fn():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True, load_in_8bit=False, compile=False, ov_config=get_default_properties())
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    models_path = pathlib.Path("cacheopt_test_models") / model_id
    models_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(models_path)
    ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True, skip_special_tokens=True)
    serialize(ov_tokenizer, models_path / "openvino_tokenizer.xml")
    serialize(ov_detokenizer, models_path / "openvino_detokenizer.xml")

    seqs_per_request = 32
    num_kv_blocks = 1000
    scheduler_config = get_scheduler_config(num_kv_blocks)

    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 100

    #scheduler_config.enable_prefix_caching = False
    model_cb = ContinuousBatchingPipeline(models_path, scheduler_config, "CPU", {}, get_default_properties())

    batch = []
    mock_prompts = ["foo bar"] * 20
    seqs_per_request = 10
    batches_processed = 0
    for p_idx, p in enumerate(mock_prompts):
        batch.append(p)
        if (
            len(batch) == seqs_per_request
            or p_idx == len(mock_prompts) - 1
        ):
            print(f"Batch {batches_processed}")
            batches_processed += 1
            _ = model_cb.generate(
                batch, [generation_config] * len(batch)
            )
            print_rss()

            batch.clear()


    del model_cb
    del model

if __name__ == "__main__":
    for i in range(100):
        print(f"Iteration {i}")
        leak_fn()
        print_rss()
