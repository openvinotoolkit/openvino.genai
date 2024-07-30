from transformers import AutoModelForCausalLM, AutoTokenizer
import whowhatbench

model_id = "facebook/opt-1.3b"
base_small = AutoModelForCausalLM.from_pretrained(model_id).cuda()
ov_model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

evaluator = whowhatbench.Evaluator(base_model=base_small, tokenizer=tokenizer)
metrics_per_prompt, metrics = evaluator.score(ov_model)

metric_of_interest = "similarity"
print(metric_of_interest, ": ", metrics["similarity"][0])

worst_examples = evaluator.worst_examples(top_k=5, metric=metric_of_interest)
print("Metric: ", metric_of_interest)
for e in worst_examples:
    print("\t=========================")
    print("\tPrompt: ", e["prompt"])
    print("\tBaseline Model:\n ", "\t" + e["source_model"])
    print("\tOptimized Model:\n ", "\t" + e["optimized_model"])
