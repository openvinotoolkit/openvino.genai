def models_list():
    model_ids = [
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B-Chat-v1.0"),
        # ("microsoft/phi-1_5", "phi-1_5/"),

        # ("google/gemma-2b-it", "gemma-2b-it"),
        # ("google/gemma-7b-it", "gemma-7b-it"),
        # ("meta-llama/Llama-2-7b-chat-hf", "Llama-2-7b-chat-hf"),
        # ("meta-llama/Llama-2-13b-chat-hf", "Llama-2-13b-chat-hf"),
        # ("openlm-research/open_llama_3b", "open_llama_3b"),
        # ("openlm-research/open_llama_7b", "open_llama_7b"),
        # ("databricks/dolly-v2-3b", "dolly-v2-3b"),
        # ("databricks/dolly-v2-12b", "dolly-v2-12b"),
    ]
    import os
    prefix = os.getenv('GENAI_MODELS_PATH_PREFIX', '')
    return [(model_id, os.path.join(prefix, model_path)) for model_id, model_path in model_ids]


if __name__ == "__main__":
    for model_id, model_path in models_list():
        print(model_id, model_path)
