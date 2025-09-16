import os
import pytest
import shutil
import logging
from utils.constants import get_ov_cache_models_dir

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown():
    """Fixture to set up and tear down the temporary directories."""

    ov_cache_models_dir = get_ov_cache_models_dir()

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
