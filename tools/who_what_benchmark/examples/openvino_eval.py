import whowhatbench
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = OVModelForCausalLM.from_pretrained(model_id, load_in_8bit=False, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)


evaluator = whowhatbench.Evaluator(base_model=model, tokenizer=tokenizer)

model_int8 = OVModelForCausalLM.from_pretrained(
    model_id, load_in_8bit=True, export=True
)
all_metrics_per_question, all_metrics = evaluator.score(model_int8)

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
