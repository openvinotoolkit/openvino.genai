import whowhatbench
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-2-7b-chat-hf"
model_gptq_id = "TheBloke/Llama-2-7B-Chat-GPTQ"

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)


evaluator = whowhatbench.Evaluator(base_model=model, tokenizer=tokenizer)

model_int4 = AutoModelForCausalLM.from_pretrained(model_gptq_id, device_map="auto")
all_metrics_per_question, all_metrics = evaluator.score(model_int4)

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
