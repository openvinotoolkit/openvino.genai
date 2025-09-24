import os
from pathlib import Path
from typing import Callable
import pytest
import shutil
import logging

from transformers import AutoTokenizer
from optimum.intel.openvino.modeling import OVModel
from tests.python_tests.utils.hugging_face import download_and_convert_model
from utils.constants import OvTestCacheManager, ModelDownloaderCallable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def ov_cache_manager(pytestconfig: pytest.Config) -> OvTestCacheManager:
    return OvTestCacheManager(pytestconfig)


@pytest.fixture(scope="session")
def ov_cache_models_dir(ov_cache_manager: OvTestCacheManager) -> Path:
    return ov_cache_manager.get_models_dir()


@pytest.fixture(scope="session")
def ov_cache_dir(ov_cache_manager: OvTestCacheManager) -> Path:
    return ov_cache_manager.get_cache_dir()


@pytest.fixture(scope="session")
def model_downloader(ov_cache_models_dir: Path) -> ModelDownloaderCallable:
    def download_model(model_id: str) -> tuple:
        return download_and_convert_model(model_id, ov_cache_models_dir)

    return download_model


@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown(ov_cache_models_dir: Path):
    """Fixture to set up and tear down the temporary directories."""
    logger.info(f"Creating directory: {ov_cache_models_dir}")
    os.makedirs(ov_cache_models_dir, exist_ok=True)

    yield

    if os.environ.get("CLEANUP_CACHE", "false").lower() != "false":
        if os.path.exists(ov_cache_models_dir):
            logger.info(f"Removing temporary directory: {ov_cache_models_dir}")
            shutil.rmtree(ov_cache_models_dir)
        else:
            logger.info(
                f"Skipped temporary directory cleanup because it doesn't exist: {ov_cache_models_dir}"
            )


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