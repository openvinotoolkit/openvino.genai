def models_list():
    model_ids = [
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B-Chat-v1.0"),
        # ("google/gemma-2b-it", "gemma-2b-it"),
        # ("google/gemma-7b-it", "gemma-7b-it"),
        # ("meta-llama/Llama-2-7b-chat-hf", "Llama-2-7b-chat-hf"),
        # ("meta-llama/Llama-2-13b-chat-hf", "Llama-2-13b-chat-hf"),
        # ("openlm-research/open_llama_3b", "open_llama_3b"),
        # ("openlm-research/open_llama_7b", "open_llama_7b"),
        # ("databricks/dolly-v2-3b", "dolly-v2-3b"),
        # ("databricks/dolly-v2-12b", "dolly-v2-12b"),
        # ("mistralai/Mistral-7B-v0.1", "Mistral-7B-v0.1"),
        # ("ikala/redpajama-3b-chat", "redpajama-3b-chat"),
        # ("microsoft/phi-1_5", "phi-1_5/"),
        # ("Qwen/Qwen1.5-7B-Chat", "Qwen1.5-7B-Chat"),
    ]
    return model_ids

if __name__ == "__main__":
    for model_id, model_path in models_list():
        print(model_id, model_path)
