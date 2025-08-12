import subprocess # nosec B404
import os
import json
import pytest
import shutil
import logging
import gc
from pathlib import Path

from utils.network import retry_request


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary containing model configurations.
# Each key is a model identifier, and the value is a dictionary with:
# - "name": the model's name or path
# - "convert_args": a list of arguments for the conversion command
MODELS = {
    "TinyLlama-1.1B-Chat-v1.0": { 
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "convert_args": ['--weight-format', 'fp16']
    },
    "SmolLM-135M": {
        "name": "HuggingFaceTB/SmolLM-135M",
        "convert_args": ['--trust-remote-code']
    },
    "SmolLM2-135M": {
        "name": "HuggingFaceTB/SmolLM2-135M",
        "convert_args": ['--trust-remote-code']
    },
    "SmolLM2-360M": {
        "name": "HuggingFaceTB/SmolLM2-360M",
        "convert_args": ['--trust-remote-code']
    },  
    "WhisperTiny": {
        "name": "openai/whisper-tiny",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16']
    },
    "Qwen2.5-0.5B-Instruct": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "convert_args": ['--trust-remote-code']
    },
    "Qwen2-0.5B-Instruct": {
        "name": "Qwen/Qwen2-0.5B-Instruct",
        "convert_args": ['--trust-remote-code']
    },
    "phi-1_5": {
        "name": "microsoft/phi-1_5",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16']
    },
    "TinyStories-1M": {
        "name": "roneneldan/TinyStories-1M",
        "convert_args": ['--trust-remote-code']
    },
    "dreamlike-anime-1.0": {
        "name": "dreamlike-art/dreamlike-anime-1.0",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16', "--task", "stable-diffusion"]
    },
    "LCM_Dreamshaper_v7-int8-ov": {
        "name": "OpenVINO/LCM_Dreamshaper_v7-int8-ov",
        "convert_args": []
    },   
    "llava-1.5-7b-hf": {
        "name": "llava-hf/llava-1.5-7b-hf",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16']
    },    
    "llava-v1.6-mistral-7b-hf": {
        "name": "llava-hf/llava-v1.6-mistral-7b-hf",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16']
    },
    "dreamlike-anime-1.0": {
        "name": "dreamlike-art/dreamlike-anime-1.0",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16', "--task", "stable-diffusion"]
    },
    "LCM_Dreamshaper_v7-int8-ov": {
        "name": "OpenVINO/LCM_Dreamshaper_v7-int8-ov",
        "convert_args": []
    },
    "tiny-random-minicpmv-2_6": {
        "name": "katuni4ka/tiny-random-minicpmv-2_6",
        "convert_args": ['--trust-remote-code', "--task", "image-text-to-text"]
    },
    "InternVL2-1B": {
        "name": "OpenGVLab/InternVL2-1B",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16']
    },
    "Qwen2-VL-2B-Instruct": {
        "name": "Qwen/Qwen2-VL-2B-Instruct",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16']
    },
    "tiny-dummy-qwen2": {
        "name": "fxmarty/tiny-dummy-qwen2",
        "convert_args": ['--trust-remote-code']
    },
    "tiny-random-qwen2": {
        "name": "fxmarty/tiny-dummy-qwen2",
        "convert_args": ["--task", "text-generation-with-past", "--weight-format", "fp16"]
    },
    "tiny-random-qwen2-int8": {
        "name": "fxmarty/tiny-dummy-qwen2",
        "convert_args": ["--task", "text-generation-with-past", "--weight-format", "int8"]
    },
    "tiny-random-latent-consistency": {
        "name": "echarlaix/tiny-random-latent-consistency",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16']
    },
    "tiny-random-latent-consistency-lora": {
        "name": "katuni4ka/tiny-random-latent-consistency-lora",
        "convert_args": []
    },
    "tiny-random-llava": {
        "name": "katuni4ka/tiny-random-llava",
        "convert_args": ["--trust-remote-code", "--task", "image-text-to-text"]
    },
    "bge-small-en-v1.5": {
        "name": "BAAI/bge-small-en-v1.5",
        "convert_args": ["--trust-remote-code"]
    },
    "ms-marco-TinyBERT-L2-v2": {
        "name": "cross-encoder/ms-marco-TinyBERT-L2-v2",
        "convert_args": ["--trust-remote-code", "--task", "text-classification"],
    },
    "tiny-random-SpeechT5ForTextToSpeech": {
        "name": "hf-internal-testing/tiny-random-SpeechT5ForTextToSpeech",
        "convert_args": ["--model-kwargs",  json.dumps({"vocoder": "fxmarty/speecht5-hifigan-tiny"})]
    },
}

SAMPLES_PY_DIR = Path(os.environ.get("SAMPLES_PY_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../samples/python"))))
SAMPLES_CPP_DIR = Path(os.environ.get("SAMPLES_CPP_DIR", os.getcwd()))
SAMPLES_C_DIR = os.environ.get("SAMPLES_C_DIR", os.getcwd())
SAMPLES_JS_DIR = os.environ.get("SAMPLES_JS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../samples/js")))


@pytest.fixture(scope="session")
def convert_model(request):
    """Fixture to convert the model once for the session."""
    models_cache = request.config.cache.get("MODELS_DIR", None)
    model_id = request.param
    model_name = MODELS[model_id]["name"]
    model_cache = os.path.join(models_cache, model_id)
    model_path = os.path.join(model_cache, model_name)
    model_args = MODELS[model_id]["convert_args"]
    logger.info(f"Preparing model: {model_name}")
    # Convert the model if not already converted
    if not os.path.exists(model_path):
        logger.info(f"Converting model: {model_name}")
        sub_env=os.environ.copy()
        command = [
            "optimum-cli", "export", "openvino",
            "--model", model_name, 
            model_path
        ]
        if model_args:
            command.extend(model_args)
        logger.info(f"Conversion command: {' '.join(command)}")
        try:
            retry_request(lambda: subprocess.run(command, check=True, text=True, env=sub_env, stderr=subprocess.STDOUT, stdout=subprocess.PIPE))
        except subprocess.CalledProcessError as error:
            logger.error(f"optimum-cli returned {error.returncode}. Output:\n{error.output}")
            raise

    yield model_path

    # Cleanup the model after tests
    if os.environ.get("CLEANUP_CACHE", "false").lower() == "true":
        if os.path.exists(model_cache):
            logger.info(f"Removing converted model: {model_cache}")
            shutil.rmtree(model_cache)

@pytest.fixture(scope="session")
def download_model(request):
    """Fixture to download the model once for the session."""
    models_cache = request.config.cache.get("MODELS_DIR", None)
    model_id = request.param
    model_name = MODELS[model_id]["name"]
    model_cache = os.path.join(models_cache, model_id)
    model_path = os.path.join(model_cache, model_name)
    logger.info(f"Preparing model: {model_name}")
    # Download the model if not already downloaded
    if not os.path.exists(model_path):
        logger.info(f"Downloading the model: {model_name}")
        sub_env=os.environ.copy()
        command = ["huggingface-cli", "download", model_name, "--local-dir", model_path]
        logger.info(f"Downloading command: {' '.join(command)}")
        retry_request(lambda: subprocess.run(command, check=True, capture_output=True, text=True, env=sub_env))
            
    yield model_path
    
    # Cleanup the model after tests
    if os.environ.get("CLEANUP_CACHE", "false").lower() == "true":
        if os.path.exists(model_cache):
            logger.info(f"Removing converted model: {model_cache}")
            shutil.rmtree(model_cache)

@pytest.fixture(scope="session")
def generate_test_content(request):
    """Generate an image of lines and return the file path."""
    
    test_data = request.config.cache.get("TEST_DATA", None)
    
    file_name = request.param
    file_path = os.path.join(test_data, file_name)
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        from PIL import Image
        import numpy as np
        res = 28, 28
        lines = np.arange(res[0] * res[1] * 3, dtype=np.uint8) % 255
        lines = lines.reshape([*res, 3])
        lines_image = Image.fromarray(lines)
        lines_image.save(file_path)
        logger.info(f"Generated test content {file_path}")
    else:
        logger.info(f"Test content already exists at {file_path}")
    yield file_path
    # Cleanup the test content after tests
    if os.environ.get("CLEANUP_CACHE", "false").lower() == "true":
        if os.path.exists(file_path):
            logger.info(f"Removing test content: {file_path}")
            os.remove(file_path)

@pytest.fixture(scope="session")
def generate_image_generation_jsonl(request):
    """Generate a JSONL file for image generation prompts."""
    
    test_data = request.config.cache.get("TEST_DATA", None)
    file_name, json_entries = request.param
    file_path = os.path.join(test_data, file_name)
    
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            for entry in json_entries:
                f.write(json.dumps(entry) + "\n")
        
        logger.info(f"Generated image generation JSONL file at {file_path}")
    else:
        logger.info(f"Image generation JSONL file already exists at {file_path}")
    
    yield file_path
    
    # Cleanup the JSONL file after tests
    if os.environ.get("CLEANUP_CACHE", "false").lower() == "true":
        if os.path.exists(file_path):
            logger.info(f"Removing JSONL file: {file_path}")
            os.remove(file_path)

@pytest.fixture(scope="module", autouse=True)
def run_gc_after_test():
    """
    Fixture to run garbage collection after each test module.
    This is a workaround to minimize memory consumption during tests and allow the use of less powerful CI runners.
    """
    yield
    gc.collect()
