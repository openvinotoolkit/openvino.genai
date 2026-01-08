import os
import gc
import pytest
import shutil
import logging
from pathlib import Path
from utils.constants import OvTestCacheManager, ModelDownloaderCallable
from utils.hugging_face import download_and_convert_model

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
    def _download_model(model_id: str, **kwargs):
        schema = download_and_convert_model(model_id, models_dir=ov_cache_models_dir, **kwargs)
        return schema.opt_model, schema.hf_tokenizer, schema.models_path
    return _download_model


@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown(ov_cache_models_dir: Path, ov_cache_dir: Path):
    logger.info(f"Creating directory: {ov_cache_models_dir}")
    ov_cache_models_dir.mkdir(exist_ok=True, parents=True)

    yield

    if os.environ.get("CLEANUP_CACHE", "false").lower() != "false":
        if ov_cache_dir.exists():
            logger.info(f"Removing temporary directory: {ov_cache_dir}")
            shutil.rmtree(ov_cache_dir)
        else:
            logger.info(f"Skipped directory cleanup because it doesn't exist: {ov_cache_dir}")


def pytest_make_parametrize_id(config, val, argname):
    if argname in ["prompt", "prompts", "batched_prompts"]:
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
    yield
    gc.collect()
