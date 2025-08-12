import os
import pytest
import shutil
import logging
import requests
from utils.constants import get_ov_cache_models_dir, get_ov_cache_dir

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    "cmu_us_awb_arctic-wav-arctic_a0001.bin": "https://huggingface.co/datasets/Xenova/cmu-arctic-xvectors-extracted/resolve/main/cmu_us_awb_arctic-wav-arctic_a0001.bin",
    "MileBench_part0.tar.gz": "https://huggingface.co/datasets/FreedomIntelligence/MileBench/resolve/main/MileBench_part0.tar.gz",
    "MileBench_part2.tar.gz": "https://huggingface.co/datasets/FreedomIntelligence/MileBench/resolve/main/MileBench_part2.tar.gz",
}


@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown(request, tmp_path_factory):
    """Fixture to set up and tear down the temporary directories."""

    ov_cache_models_dir = get_ov_cache_models_dir()

    logger.info(f"Creating directory: {ov_cache_models_dir}")
    os.makedirs(ov_cache_models_dir, exist_ok=True)

    ov_cache = get_ov_cache_dir(tmp_path_factory.mktemp("ov_cache"))
    models_dir = os.path.join(ov_cache, "test_models")
    hf_home = os.environ.get("HF_HOME", ov_cache)
    test_data = os.path.normpath(os.path.join(hf_home, "test_data"))

    logger.info(f"Creating directories: {models_dir} and {test_data}")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(test_data, exist_ok=True)

    request.config.cache.set("OV_CACHE", str(ov_cache))
    request.config.cache.set("MODELS_DIR", str(models_dir))
    request.config.cache.set("TEST_DATA", str(test_data))

    yield

    if os.environ.get("CLEANUP_CACHE", "false").lower() != "false":
        if os.path.exists(ov_cache_models_dir):
            logger.info(f"Removing temporary directory: {ov_cache_models_dir}")
            shutil.rmtree(ov_cache_models_dir)
        else:
            logger.info(
                f"Skipped temporary directory cleanup because it doesn't exist: {ov_cache_models_dir}"
            )
        if os.path.exists(ov_cache):
            logger.info(f"Removing temporary directory: {ov_cache}")
            shutil.rmtree(ov_cache)
        else:
            logger.info(f"Skipping cleanup of temporary directory: {ov_cache}")


def pytest_make_parametrize_id(config, val, argname):
    if argname in ["prompt", "prompts", "batched_prompts"]:
        # Print only first 1000 characters of long prompts.
        if isinstance(val, list):
            return ", ".join([f"{v[:100]}" for v in val])
        else:
            return f"{val[:100]}"
    elif argname == "model_descr":
        return f"{val[0]}"
    elif argname == "chat_config":
        return f"{val[0]}"
    elif argname in ["stop_criteria", "generation_config"]:
        return str(val)
    elif isinstance(val, (int, float, str)):
        return f"{argname}={val}"
    return None


def pytest_addoption(parser):
    parser.addoption("--model_ids", help="Select models to run")


def pytest_configure(config: pytest.Config):
    marker = "precommit" if config.getoption("-m") == "precommit" else None
    pytest.run_marker = marker
    pytest.selected_model_ids = config.getoption("--model_ids", default=None)


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
        file_name_with_tar = os.path.splitext(file_name)[0]  # Removes .gz
        file_name = os.path.splitext(file_name_with_tar)[0]  # Removes .tar
        extracted_dir = os.path.join(test_data, file_name)
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

download_mask_image = download_test_content
