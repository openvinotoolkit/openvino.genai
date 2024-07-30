from pathlib import PosixPath
import os

import whowhatbench
from whowhatbench.wwb import load_dataset
from optimum.intel.openvino import OVModelForCausalLM

from openvino_genai import ContinuousBatchingPipeline, SchedulerConfig, GenerationResult, GenerationConfig

from openvino_tokenizers import convert_tokenizer
from openvino import serialize
from transformers import AutoTokenizer

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 128
SEQS_PER_REQUEST = 5
MAX_SEQUENCES = 100


model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_path = PosixPath("/tmp/models_wwb") / model_id
model.save_pretrained(model_path)

ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True, skip_special_tokens=True)
serialize(ov_tokenizer, model_path / "openvino_tokenizer.xml")
serialize(ov_detokenizer, model_path / "openvino_detokenizer.xml")

scheduler_config = SchedulerConfig()
scheduler_config.num_kv_blocks = 300
scheduler_config.dynamic_split_fuse = True
scheduler_config.max_num_batched_tokens = 256
scheduler_config.max_num_seqs = 256
scheduler_config.use_cache_eviction = True

generation_config = GenerationConfig()
generation_config.num_return_sequences = 1
generation_config.max_new_tokens = MAX_NEW_TOKENS

data = load_dataset(path='squad', name=None, split='validation')["context"]
data_dict = {"questions": list(data)[:MAX_SEQUENCES]}

model_cb = pipe = ContinuousBatchingPipeline(model_path.absolute().as_posix(), scheduler_config, "CPU", {})


GT_DATA_FILE = 'gt_data.csv'

if os.path.exists(GT_DATA_FILE):
    evaluator = whowhatbench.Evaluator(base_model=None, gt_data=GT_DATA_FILE, tokenizer=tokenizer, test_data=data_dict, generation_config=generation_config, max_new_tokens=MAX_NEW_TOKENS, seqs_per_request=3)
else:
    evaluator = whowhatbench.Evaluator(base_model=model, tokenizer=tokenizer, test_data=data_dict, generation_config=generation_config, max_new_tokens=MAX_NEW_TOKENS, seqs_per_request=3)
    evaluator.dump_gt('gt_data.csv')

all_metrics_per_question, all_metrics = evaluator.score(model_cb)

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
