import subprocess # nosec B404
import os
import tempfile
import pytest
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define model names and directories
MODELS = {
    "TinyLlama-1.1B-Chat-v1.0": { 
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "convert_args": []
    },
    "TinyLlama-1.1B-intermediate-step-1431k-3T": {
        "name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "convert_args": ['--trust-remote-code']
    },  
    "WhisperTiny": {
        "name": "openai/whisper-tiny",
        "convert_args": ['--trust-remote-code']
    },
    "open_llama_3b_v2": {
        "name": "openlm-research/open_llama_3b_v2",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16']
    },
    "Qwen-7B-Chat": {
        "name": "Qwen/Qwen-7B-Chat",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16']
    },
    "Qwen1.5-7B-Chat": {
        "name": "Qwen/Qwen1.5-7B-Chat",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16']
    },
    "phi-2": {
        "name": "microsoft/phi-2",
        "convert_args": ['--trust-remote-code', '--weight-format', 'fp16']
    }
}

TEST_FILES = {
    "how_are_you_doing_today.wav": "https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav",
    "adapter_model.safetensors": "https://huggingface.co/smangrul/tinyllama_lora_sql/resolve/main/adapter_model.safetensors"
}

TEMP_DIR = os.environ.get("TEMP_DIR", tempfile.mkdtemp())
MODELS_DIR = os.path.join(TEMP_DIR, "test_models")
TEST_DATA = os.path.join(TEMP_DIR, "test_data")

SAMPLES_PY_DIR = os.environ.get("SAMPLES_PY_DIR", os.getcwd())
SAMPLES_CPP_DIR = os.environ.get("SAMPLES_CPP_DIR", os.getcwd())

# A shared fixture to hold data
@pytest.fixture(scope="session")
def shared_data():
    return {}

@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown(request):
    """Fixture to set up and tear down the temporary directories."""
    logger.info(f"Creating directories: {MODELS_DIR} and {TEST_DATA}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TEST_DATA, exist_ok=True)
    yield
    if not os.environ.get("TEMP_DIR"):
        logger.info(f"Removing temporary directory: {TEMP_DIR}")
        shutil.rmtree(TEMP_DIR)
    else:
        logger.info(f"Skipping cleanup of temporary directory: {TEMP_DIR}")

@pytest.fixture(scope="session")
def convert_model(request):
    """Fixture to convert the model once for the session."""
    model_id = request.param
    model_name = MODELS[model_id].get("name")
    model_path = os.path.join(MODELS_DIR, model_name)
    model_args = MODELS[model_id].get("convert_args", [])
    logger.info(f"Preparing model: {model_name}")
    # Convert the model if not already converted
    if not os.path.exists(model_path):
        logger.info(f"Converting model: {model_name}")
        command = [
            "optimum-cli", "export", "openvino",
            "--model", model_name, model_path
        ]
        if model_args:
            command.extend(model_args)
        result = subprocess.run(command, check=True)
        assert result.returncode == 0, f"Model {model_name} conversion failed"
    yield model_path
    # Cleanup the model after tests
    if not os.environ.get("TEMP_DIR"):
        if os.path.exists(model_path):
            logger.info(f"Removing converted model: {model_path}")
            shutil.rmtree(model_path)
    else:
        logger.info(f"The model {model_path} will be removed when all the tests are finished")

@pytest.fixture(scope="session")
def download_test_content(request):
    """Download the test content from the given URL and return the file path."""
    file_url = request.param
    file_name = os.path.basename(file_url)
    file_path = os.path.join(TEST_DATA, file_name)
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
    if os.path.exists(file_path):
        logger.info(f"Removing test content: {file_path}")
        os.remove(file_path)