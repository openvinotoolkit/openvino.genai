import subprocess # nosec B404
import os
import json
import pytest
import shutil
import logging
import gc
import requests
from pathlib import Path

from utils.network import retry_request
from utils.constants import get_ov_cache_dir


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
    "SmolLM2-135M-GGUF": {
        "name": "prithivMLmods/SmolLM2-135M-GGUF",
        "gguf_filename": "SmolLM2-135M.F16.gguf",
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
    "Qwen2.5-0.5B-Instruct-GGUF": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "gguf_filename": "qwen2.5-0.5b-instruct-q4_0.gguf",
        "convert_args": ['--trust-remote-code']
    },
    "Qwen2-0.5B-Instruct": {
        "name": "Qwen/Qwen2-0.5B-Instruct",
        "convert_args": ['--trust-remote-code']
    },
    "Qwen2-0.5B-Instruct-GGUF": {
        "name": "Qwen/Qwen2-0.5B-Instruct-GGUF",
        "gguf_filename": "qwen2-0_5b-instruct-q4_0.gguf",
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
    }
}

TEST_FILES = {
    "how_are_you_doing_today.wav": "https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav",
    "adapter_model.safetensors": "https://huggingface.co/smangrul/tinyllama_lora_sql/resolve/main/adapter_model.safetensors",
    "monalisa.jpg": "https://llava-vl.github.io/static/images/monalisa.jpg",
    "soulcard.safetensors": "https://civitai.com/api/download/models/72591",
    "ShareGPT_V3_unfiltered_cleaned_split.json": "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
    "images/image.png": "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
    "mask_image.png": "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png",
    "overture-creations.png": "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
    "overture-creations-mask.png": "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png",
    "cat.png": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
    "cat": "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11",
    "3283_1447_000.tar.gz": "https://huggingface.co/datasets/facebook/multilingual_librispeech/resolve/main/data/mls_polish/train/audio/3283_1447_000.tar.gz",
    "cmu_us_awb_arctic-wav-arctic_a0001.bin": "https://huggingface.co/datasets/Xenova/cmu-arctic-xvectors-extracted/resolve/main/cmu_us_awb_arctic-wav-arctic_a0001.bin"
}

SAMPLES_PY_DIR = Path(os.environ.get("SAMPLES_PY_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../samples/python"))))
SAMPLES_CPP_DIR = Path(os.environ.get("SAMPLES_CPP_DIR", os.getcwd()))
SAMPLES_C_DIR = os.environ.get("SAMPLES_C_DIR", os.getcwd())
SAMPLES_JS_DIR = os.environ.get("SAMPLES_JS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../samples/js")))

@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown(request, tmp_path_factory):
    """Fixture to set up and tear down the temporary directories."""
    
    ov_cache = get_ov_cache_dir(tmp_path_factory.mktemp("ov_cache"))  
    models_dir = os.path.join(ov_cache, "test_models")
    test_data = os.path.join(ov_cache, "test_data")
    
    logger.info(f"Creating directories: {models_dir} and {test_data}")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(test_data, exist_ok=True)
    
    request.config.cache.set("OV_CACHE", str(ov_cache))
    request.config.cache.set("MODELS_DIR", str(models_dir))
    request.config.cache.set("TEST_DATA", str(test_data))
    
    yield
    
    if os.environ.get("CLEANUP_CACHE", "false").lower() != "false":
        if os.path.exists(ov_cache):
            logger.info(f"Removing temporary directory: {ov_cache}")
            shutil.rmtree(ov_cache)
        else:
            logger.info(f"Skipping cleanup of temporary directory: {ov_cache}")


def download_gguf_model(model, model_path):
    """Download the GGUF model using huggingface-cli."""
    sub_env = os.environ.copy()
    model_name = model["name"]
    model_gguf_filename = model["gguf_filename"]
    command = ["huggingface-cli", "download", model_name, model_gguf_filename, "--local-dir", model_path]
    logger.info(f"Downloading command: {' '.join(command)}")
    try:
        retry_request(lambda: subprocess.run(command, check=True, text=True, env=sub_env, stderr=subprocess.STDOUT, stdout=subprocess.PIPE))
    except subprocess.CalledProcessError as error:
        logger.error(f"huggingface-cli returned {error.returncode}. Output:\n{error.output}")
        raise

def optimum_cli_convert(model, model_path):
    """Convert the model using optimum-cli."""
    sub_env = os.environ.copy()
    model_name = model["name"]
    model_args = model["convert_args"]
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

@pytest.fixture(scope="session")
def convert_model(request):
    """Fixture to convert the model once for the session."""
    models_cache = request.config.cache.get("MODELS_DIR", None)
    model_id = request.param
    model = MODELS[model_id]
    model_name = model["name"]
    model_cache = os.path.join(models_cache, model_id)
    model_path = os.path.join(model_cache, model_name)
    logger.info(f"Preparing model: {model_name}")
    if not os.path.exists(model_path):
        if "gguf_filename" in model:
            # Download the GGUF model if not already downloaded
            download_gguf_model(model, model_path)
        else:
            # Convert the model if not already converted
            optimum_cli_convert(model, model_path)

    if "gguf_filename" in model:
        model_path = os.path.join(model_path, model["gguf_filename"])
    yield model_path

    # Cleanup the model after tests
    if os.environ.get("CLEANUP_CACHE", "false").lower() == "true":
        if os.path.exists(model_cache):
            logger.info(f"Removing cached model: {model_cache}")
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
def download_test_content(request):
    """Download the test content from the given URL and return the file path or extracted folder."""
    
    test_data = request.config.cache.get("TEST_DATA", None)
    
    file_name = request.param
    file_url = TEST_FILES[file_name]
    file_path = os.path.join(test_data, file_name)
    
    if not os.path.exists(file_path):
        logger.info(f"Downloading test content from {file_url} to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        response = requests.get(file_url, stream=True)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Downloaded test content to {file_path}")
    else:
        logger.info(f"Test content already exists at {file_path}")

    # If the file is a tarball, extract it
    extracted_dir = None
    if file_name.endswith(".tar.gz"):
        extracted_dir = os.path.join(test_data, os.path.splitext(file_name)[0])
        if not os.path.exists(extracted_dir):
            os.makedirs(extracted_dir, exist_ok=True)
            shutil.unpack_archive(file_path, extracted_dir)
            logger.info(f"Extracted tarball to {extracted_dir}")
        else:
            logger.info(f"Extracted folder already exists at {extracted_dir}")
        yield extracted_dir
    else:
        yield file_path

    # Cleanup the test content after tests
    if os.environ.get("CLEANUP_CACHE", "false").lower() == "true":
        if extracted_dir and os.path.exists(extracted_dir):
            logger.info(f"Removing extracted folder: {extracted_dir}")
            shutil.rmtree(extracted_dir)
        if os.path.exists(file_path):
            logger.info(f"Removing test content: {file_path}")
            os.remove(file_path)


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
