import whowhatbench
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__


def get_transformers_quantization_config() -> dict:
    quantization_config = {}

    transformers_version = Version(__version__)
    if transformers_version < Version("5.0.0"):
        quantization_config = {"load_in_4bit": True}
    else:
        from transformers import BitsAndBytesConfig

        quantization_config["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    return quantization_config


model_id = "meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

evaluator = whowhatbench.Evaluator(base_model=model, tokenizer=tokenizer)

quantization_config = get_transformers_quantization_config()
model_int4 = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", **quantization_config)
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
