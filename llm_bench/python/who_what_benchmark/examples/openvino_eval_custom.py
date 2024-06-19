from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
import whowhatbench
from datasets import load_dataset

model_id_base = "/nfs/ov-share-05/data/cv_bench_cache/DL_benchmarking_models/baichuan2-13b-chat/pytorch"
model_id_ov = "/nfs/ov-share-05/data/cv_bench_cache/WW24_llm_2024.3.0-15670-98180edcbcc/baichuan2-13b-chat/pytorch/dldt/FP16"

trust_remote_code = True
base_small = AutoModelForCausalLM.from_pretrained(model_id_base, trust_remote_code=trust_remote_code)
optimized_model = OVModelForCausalLM.from_pretrained(model_id_ov, trust_remote_code=trust_remote_code)
tokenizer = AutoTokenizer.from_pretrained(model_id_base, trust_remote_code=trust_remote_code)

val = load_dataset("wikitext",'wikitext-103-raw-v1',  split="validation[:2]")
prompts = val["text"]

evaluator = whowhatbench.Evaluator(base_model=base_small, tokenizer=tokenizer)
metrics_per_prompt, all_metrics = evaluator.score(optimized_model, test_data=prompts)
print(f"\t=========================  ALL METRICS =========================")
print(metrics_per_prompt)
print(all_metrics)
print(f"\t=========================  ALL METRICS =========================")

metrics = ["similarity", "FDT", "FDT norm", "SDT", "SDT norm"]

for metric in metrics:
    worst_examples = evaluator.worst_examples(top_k=5, metric=metric)
    print("Metric: ", metric)
    for idx, e in enumerate(worst_examples):
        print(f"\t========================= {metric} - {idx}")
        print("\tPrompt: ", e["prompt"])
        print("\tBaseline Model:\n ", "\t" + e["source_model"])
        print("\tOptimized Model:\n ", "\t" + e["optimized_model"])
