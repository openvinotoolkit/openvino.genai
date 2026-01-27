import os
import gc
import pytest
import shutil
import logging
from utils.constants import (
    get_ov_cache_dir,
    get_ov_cache_downloaded_models_dir,
    get_ov_cache_converted_models_dir,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown():
    """Fixture to set up and tear down the temporary directories."""

    ov_cache_dir = get_ov_cache_dir()
    ov_cache_downloaded_dir = get_ov_cache_downloaded_models_dir()
    ov_cache_converted_dir = get_ov_cache_converted_models_dir()

    logger.info(f"Creating directories: {ov_cache_downloaded_dir}, {ov_cache_converted_dir}")
    ov_cache_downloaded_dir.mkdir(exist_ok=True, parents=True)
    ov_cache_converted_dir.mkdir(exist_ok=True, parents=True)

    yield

    if os.environ.get("CLEANUP_CACHE", "false").lower() != "false":
        if ov_cache_dir.exists():
            logger.info(f"Removing temporary directory: {ov_cache_dir}")
            shutil.rmtree(ov_cache_dir)
        else:
            logger.info(
                f"Skipped temporary directory cleanup because it doesn't exist: {ov_cache_dir}"
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
    pytest.selected_model_ids = config.getoption("--model_ids", default=None)


@pytest.fixture(scope="module", autouse=True)
def run_gc_after_test():
    """
    Fixture to run garbage collection after each test module.
    This is a workaround to minimize memory consumption during tests 
    and allow the use of less powerful CI runners.
    """
    yield
    gc.collect()
