from pathlib import PosixPath
import os
import tempfile

import whowhatbench
from whowhatbench.wwb import load_dataset
from optimum.intel.openvino import OVModelForCausalLM

from openvino_genai import (
    ContinuousBatchingPipeline,
    SchedulerConfig,
    GenerationConfig,
    CacheEvictionConfig,
    AggregationMode,
)

from openvino_tokenizers import convert_tokenizer
from openvino import serialize
from transformers import AutoTokenizer

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 128
SEQS_PER_REQUEST = 5
MAX_SEQUENCES = 100

model = OVModelForCausalLM.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_path = PosixPath(tempfile.gettempdir()) / model_id
model.save_pretrained(model_path)

ov_tokenizer, ov_detokenizer = convert_tokenizer(
    tokenizer, with_detokenizer=True, skip_special_tokens=True
)
serialize(ov_tokenizer, model_path / "openvino_tokenizer.xml")
serialize(ov_detokenizer, model_path / "openvino_detokenizer.xml")

scheduler_config_noopt = SchedulerConfig()
scheduler_config_noopt.num_kv_blocks = 300
scheduler_config_noopt.dynamic_split_fuse = True
scheduler_config_noopt.max_num_batched_tokens = 256
scheduler_config_noopt.max_num_seqs = 256
scheduler_config_noopt.enable_prefix_caching = False

scheduler_config_opt = SchedulerConfig()
scheduler_config_opt.num_kv_blocks = 300
scheduler_config_opt.dynamic_split_fuse = True
scheduler_config_opt.max_num_batched_tokens = 256
scheduler_config_opt.max_num_seqs = 256
scheduler_config_opt.use_cache_eviction = True
scheduler_config_opt.enable_prefix_caching = False
eviction_config = CacheEvictionConfig(32, 32, 128, AggregationMode.NORM_SUM)
scheduler_config_opt.cache_eviction_config = eviction_config

generation_config = GenerationConfig()
generation_config.num_return_sequences = 1
generation_config.max_new_tokens = MAX_NEW_TOKENS

data = load_dataset(path="squad", name=None, split="validation")["context"]
data_dict = {"prompts": list(dict({k: None for k in data}).keys())[:MAX_SEQUENCES]}

model_cb_noopt = ContinuousBatchingPipeline(
    model_path.absolute().as_posix(), scheduler_config_noopt, "CPU", {}
)
model_cb_opt = ContinuousBatchingPipeline(
    model_path.absolute().as_posix(), scheduler_config_opt, "CPU", {}
)


GT_DATA_FILE = "gt_data.csv"

if os.path.exists(GT_DATA_FILE):
    evaluator = whowhatbench.TextEvaluator(
        base_model=model_cb_noopt,
        gt_data=GT_DATA_FILE,
        tokenizer=tokenizer,
        test_data=data_dict,
        generation_config=generation_config,
        max_new_tokens=MAX_NEW_TOKENS,
        seqs_per_request=3,
    )
else:
    evaluator = whowhatbench.TextEvaluator(
        base_model=model_cb_noopt,
        tokenizer=tokenizer,
        test_data=data_dict,
        generation_config=generation_config,
        max_new_tokens=MAX_NEW_TOKENS,
        seqs_per_request=3,
    )
    evaluator.dump_gt("gt_data.csv")


all_metrics_per_question, all_metrics = evaluator.score(model_cb_opt)


print(all_metrics_per_question)
print(all_metrics)

metrics = ["similarity", "SDT norm"]

for metric in metrics:
    worst_examples = evaluator.worst_examples(top_k=5, metric=metric)
    print("Metric: ", metric)
    for e in worst_examples:
        print("\t=========================")
        print(f"\t{metric}: ", e[metric])
        print("\tPrompt: ", e["prompt"])
        print("\tSource Model:\n ", "\t" + e["source_model"])
        print("\tOptimized Model:\n ", "\t" + e["optimized_model"])

pipeline_opt_metrics = model_cb_opt.get_metrics()
pipeline_noopt_metrics = model_cb_noopt.get_metrics()

print(
    f"No-opt cache usage: max {pipeline_noopt_metrics.max_cache_usage:.3f}, avg {pipeline_noopt_metrics.avg_cache_usage:.3f}"
)
print(
    f"Opt cache usage: max {pipeline_opt_metrics.max_cache_usage:.3f}, avg {pipeline_opt_metrics.avg_cache_usage:.3f}"
)
max_optimization_ratio = (
    pipeline_noopt_metrics.max_cache_usage / pipeline_opt_metrics.max_cache_usage
)
avg_optimization_ratio = (
    pipeline_noopt_metrics.avg_cache_usage / pipeline_opt_metrics.avg_cache_usage
)
print(
    f"Optimization ratios: max {max_optimization_ratio:.3f}x, avg {avg_optimization_ratio:.3f}x"
)
