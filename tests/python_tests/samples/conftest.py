import subprocess # nosec B404
import os
import pytest
import shutil
import logging
import gc

from utils.network import retry_request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary containing model configurations.
# Each key is a model identifier, and the value is a dictionary with:
# - "name": the model's name or path
# - "convert_args": a list of arguments for the conversion command
MODELS = {
    "LaMini-GPT-124M": { 
        "name": "MBZUAI/LaMini-GPT-124M",
        "convert_args": []
    },
    "SmolLM-135M": {
        "name": "HuggingFaceTB/SmolLM-135M",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16']
    },
    "SmolLM2-135M": {
        "name": "HuggingFaceTB/SmolLM2-135M",
        "convert_args": ['--trust-remote-code']
    },  
    "WhisperTiny": {
        "name": "openai/whisper-tiny",
        "convert_args": ['--trust-remote-code']
    },
    "Qwen2.5-0.5B-Instruct": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16']
    },
    "Qwen2-0.5B-Instruct": {
        "name": "Qwen/Qwen2-0.5B-Instruct",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16']
    },
    "phi-1_5": {
        "name": "microsoft/phi-1_5",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16']
    },
    "TinyStories-1M": {
        "name": "roneneldan/TinyStories-1M",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16']
    }
}

TEST_FILES = {
    "how_are_you_doing_today.wav": "https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav",
    "adapter_model.safetensors": "https://huggingface.co/smangrul/tinyllama_lora_sql/resolve/main/adapter_model.safetensors"
}

SAMPLES_PY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../samples/python"))
SAMPLES_CPP_DIR = os.environ.get("SAMPLES_CPP_DIR", os.getcwd())

def pytest_addoption(parser):
    parser.addoption(
        "--cleanup-cache", action="store_true", default=False, help="Cleanup cache after tests"
    )

@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown(request, tmp_path_factory):
    """Fixture to set up and tear down the temporary directories."""
    
    ov_cache = os.environ.get("OV_CACHE", tmp_path_factory.mktemp("ov_cache"))
    models_dir = os.path.join(ov_cache, "test_models")
    test_data = os.path.join(ov_cache, "test_data")
    
    logger.info(f"Creating directories: {models_dir} and {test_data}")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(test_data, exist_ok=True)
    
    request.config.cache.set("OV_CACHE", str(ov_cache))
    request.config.cache.set("MODELS_DIR", str(models_dir))
    request.config.cache.set("TEST_DATA", str(test_data))
    
    yield
    
    if request.config.getoption("--cleanup-cache"):
        if os.path.exists(ov_cache):
            logger.info(f"Removing temporary directory: {ov_cache}")
            shutil.rmtree(ov_cache)
        else:
            logger.info(f"Skipping cleanup of temporary directory: {ov_cache}")

@pytest.fixture(scope="session")
def convert_model(request):
    """Fixture to convert the model once for the session."""
    models_cache = request.config.cache.get("MODELS_DIR", None)
    model_id = request.param
    if model_id not in MODELS:
        raise ValueError(f"Model ID '{model_id}' not found in MODELS dictionary.")
    model_name = MODELS[model_id]["name"]
    model_cache = os.path.join(models_cache, model_id)
    model_path = os.path.join(model_cache, model_name)
    model_args = MODELS[model_id].get("convert_args", [])
    model_hf_cache = os.environ.get("HF_HOME", os.path.join(model_cache, "hf_cache"))
    logger.info(f"Preparing model: {model_name}")
    # Convert the model if not already converted
    if not os.path.exists(model_path):
        logger.info(f"Converting model: {model_name}")
        command = [
            "optimum-cli", "export", "openvino",
            "--model", model_name, 
            "--cache_dir", model_hf_cache, 
            model_path
        ]
        if model_args:
            command.extend(model_args)
        logger.info(f"Conversion command: {' '.join(command)}")
        retry_request(lambda: subprocess.run(command, check=True))
            
    yield model_path
    
    # Cleanup the model after tests
    if request.config.getoption("--cleanup-cache"):
        if os.path.exists(model_cache):
            logger.info(f"Removing converted model: {model_cache}")
            shutil.rmtree(model_cache)

@pytest.fixture(scope="session")
def download_test_content(request):
    """Download the test content from the given URL and return the file path."""
    
    test_data = request.config.cache.get("TEST_DATA", None)
    
    file_url = request.param
    file_name = os.path.basename(file_url)
    file_path = os.path.join(test_data, file_name)
    if not os.path.exists(file_path):
        logger.info(f"Downloading test content from {file_url}...")
        result = subprocess.run(
            ["wget", file_url, "-O", file_path],
            check=True
        )
        assert result.returncode == 0, "Failed to download test content"
        logger.info(f"Downloaded test content to {file_path}")
    else:
        logger.info(f"Test content already exists at {file_path}")
    yield file_path
    # Cleanup the test content after tests
    if request.config.getoption("--cleanup-cache"):
        if os.path.exists(file_path):
            logger.info(f"Removing test content: {file_path}")
            os.remove(file_path)


@pytest.fixture(scope="module", autouse=True)
def run_gc_after_test():
    """
    Fixture to run garbage collection after each test module.
    This is a workaround to minimize memory consumption during tests and allow the use of less powerful CI runners.
    """
    yield
    gc.collect()