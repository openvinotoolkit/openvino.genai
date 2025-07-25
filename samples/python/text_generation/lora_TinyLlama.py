from pathlib import Path

import openvino_genai as ov_genai

import json
import argparse
import hashlib

def get_config(config):
    if Path(config).is_file():
        with open(config, 'r') as f:
            try:
                ov_config = json.load(f)
            except Exception:
                raise RuntimeError(f'==Parse file:{config} failure, json format is incorrect ==')
    else:
        try:
            ov_config = json.loads(config)
        except Exception:
            raise RuntimeError(f'==Parse config:{config} failure, json format is incorrect ==')
    return ov_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('device')
    parser.add_argument('config_json', nargs='?', default=None, help='Optional path to the configuration JSON file')
    args = parser.parse_args()

    model_path = args.model_path
    device = args.device
    if args.config_json is not None:
        ov_config = get_config(args.config_json)
    else:
        ov_config = None

    green_text = "\033[32m"
    reset_text = "\033[0m"

    print(f"{green_text}Step 1. Create pipeline and generate results via OpenVINO GenAI without LoRA{reset_text}")
    pipe = ov_genai.LLMPipeline(model_path, device)

    results = pipe.generate("Give me a sky blue color.", max_new_tokens=256)

    results_md5 =(hashlib.new("md5", results.encode(), usedforsecurity=False).hexdigest())

    print(results)

    print(results_md5)

    print(f"{green_text}Step 2. Create pipeline and generate results via OpenVINO GenAI with LoRA{reset_text}")
    print(f"{green_text}Step 2.1 Load adapter{reset_text}")
    from huggingface_hub import hf_hub_download

    lora_dir = Path("lora")

    colorista_lora_id = "Javascript/tinyllama-colorist-lora"
    colorita_lora_path = lora_dir / "tinyllama-colorist-lora"

    if not colorita_lora_path.exists():
        hf_hub_download(repo_id=colorista_lora_id, filename="adapter_model.safetensors", local_dir=colorita_lora_path)

    print(f"{green_text}Step 2.2 Initialize pipeline with adapters and run inference{reset_text}")
    adapter_config = ov_genai.AdapterConfig()

    colorist_adapter = ov_genai.Adapter(colorita_lora_path / "adapter_model.safetensors")
    adapter_config.add(colorist_adapter, alpha=0.75)

    if ov_config is None:
        pipe_with_adapters = ov_genai.LLMPipeline(model_path, device, adapters=adapter_config)
    else:
        pipe_with_adapters = ov_genai.LLMPipeline(model_path, device, ov_config, adapters=adapter_config)

    results = pipe_with_adapters.generate("Give me a sky blue color.", max_new_tokens=256)

    results_md5 =(hashlib.new("md5", results.encode(), usedforsecurity=False).hexdigest())

    print(results)

    print(results_md5)

    print(f"{green_text}Step 2.3 Get information about adapters{reset_text}")
    print("Loaded adapters numers: ", len(adapter_config.get_adapters()))
    print("Alpha for colorist adapter: ", adapter_config.get_alpha(colorist_adapter))

    adapter_config.get_adapters()

    print(f"{green_text}Step 2.4 Disable adapters{reset_text}")
    results = pipe_with_adapters.generate("Give me a sky blue color.", max_new_tokens=256, adapters=ov_genai.AdapterConfig())
    results_md5 =(hashlib.new("md5", results.encode(), usedforsecurity=False).hexdigest())
    print(results)
    print(results_md5)

    print(f"{green_text}Step 2.5 Remove adapters{reset_text}")
    adapter_config.remove(colorist_adapter)
    print("Loaded adapters: ", len(adapter_config.get_adapters()))

    chatbot_lora_id = "snshrivas10/sft-tiny-chatbot"
    chatbot_lora_path = lora_dir / "sft-tiny-chatbot"

    if not chatbot_lora_path.exists():
        hf_hub_download(repo_id=chatbot_lora_id, filename="adapter_model.safetensors", local_dir=chatbot_lora_path)

    med_lora_id = "therealcyberlord/TinyLlama-1.1B-Medical"
    med_lora_path = lora_dir / "TinyLlama-1.1B-Medical"

    if not med_lora_path.exists():
        hf_hub_download(repo_id=med_lora_id, filename="adapter_model.safetensors", local_dir=med_lora_path)

    chatbot_adapter = ov_genai.Adapter(chatbot_lora_path / "adapter_model.safetensors")
    adapter_config.add(chatbot_adapter)

    print("Loaded adapters: ", len(adapter_config.get_adapters()))
    print("Alpha for chatbot adapter: ", adapter_config.get_alpha(chatbot_adapter))

    print(f"{green_text}Step 2.6 Selection specific adapter during generation{reset_text}")
    results = pipe_with_adapters.generate("Give me a sky blue color.", max_new_tokens=256, adapters=adapter_config)
    results_md5 =(hashlib.new("md5", results.encode(), usedforsecurity=False).hexdigest())
    print(results)
    print(results_md5)

    print(f"{green_text}Step 2.7 Use several adapters{reset_text}")
    med_adapter = ov_genai.Adapter(med_lora_path / "adapter_model.safetensors")
    adapter_config.add(med_adapter)

    print("Loaded adapters: ", len(adapter_config.get_adapters()))
    print("Alpha for medprob-anatomy_lora adapter: ", adapter_config.get_alpha(med_adapter))

    results = pipe_with_adapters.generate("What is the structure of the frontal lobe of the brain?", max_new_tokens=256, adapters=adapter_config)
    results_md5 =(hashlib.new("md5", results.encode(), usedforsecurity=False).hexdigest())
    print(results)
    print(results_md5)

if '__main__' == __name__:
    main()
