import os
import pytest
import logging
import shutil
import subprocess  # nosec B404

from pathlib import Path
from typing import Any, Dict

from ov_utils import AtomicDownloadManager, get_ov_cache_dir, retry_request  # noqa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODELS: Dict[str, Dict[str, Any]] = {
    "bge-small-en-v1.5": {
        "name": "BAAI/bge-small-en-v1.5",
        "convert_args": ["--trust-remote-code", "--task", "feature-extraction"],
    },
    "Qwen3-Embedding-0.6B": {
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "convert_args": ["--trust-remote-code", "--task", "feature-extraction"],
    },
    "ms-marco-TinyBERT-L2-v2": {
        "name": "cross-encoder/ms-marco-TinyBERT-L2-v2",
        "convert_args": ["--trust-remote-code", "--task", "text-classification"],
    },
    "Qwen3-Reranker-0.6B": {
        "name": "Qwen/Qwen3-Reranker-0.6B",
        "convert_args": ["--trust-remote-code", "--task", "text-generation"],
    },
    "tiny-random-qwen2vl": {
        "name": "optimum-intel-internal-testing/tiny-random-qwen2vl",
        "convert_args": ["--trust-remote-code", "--task", "image-text-to-text"],
    },
    "tiny-random-llava-next-video": {
        "name": "optimum-intel-internal-testing/tiny-random-llava-next-video",
        "convert_args": ["--trust-remote-code", "--task", "image-text-to-text"],
    },
    "nanoLLaVA": {
        "name": "qnguyen3/nanoLLaVA",
        "convert_args": ["--trust-remote-code", "--task", "image-text-to-text"],
    },
    "tiny-random-llava": {
        "name": "optimum-intel-internal-testing/tiny-random-llava",
        "convert_args": ["--trust-remote-code", "--task", "image-text-to-text"],
    },
    "tiny-random-stable-diffusion-xl": {"name": "echarlaix/tiny-random-stable-diffusion-xl", "convert_args": []},
    "stable-diffusion-3-tiny-random": {"name": "yujiepan/stable-diffusion-3-tiny-random", "convert_args": []},
    "tiny-random-flux": {"name": "optimum-intel-internal-testing/tiny-random-flux", "convert_args": []},
    "tiny-random-flux-fill": {"name": "optimum-intel-internal-testing/tiny-random-flux-fill", "convert_args": []},
}


def get_ov_cache_converted_models_dir():
    return get_ov_cache_dir() / "converted_models"


def convert_model(model_name: str) -> str:
    models_dir = get_ov_cache_converted_models_dir()

    parts = model_name.split("/")
    if len(parts) != 2 or parts[1] not in MODELS:
        raise ValueError(
            f"Invalid model name '{model_name}', potentially it is not registered in MODELS set or the name is incorrect."
        )
    model_id = parts[1]

    convert_args = MODELS[model_id]["convert_args"]
    model_path = Path(models_dir) / f"wwb_{model_id}"

    manager = AtomicDownloadManager(model_path)

    logger.info(f"Start convertion of: {model_name}")
    if manager.is_complete():
        logger.info("Conversion command is already completed")
        return model_path

    def convert(temp_path: Path) -> None:
        command = [
            "optimum-cli",
            "export",
            "openvino",
            "--model",
            model_name,
            "--weight-format",
            "fp16",
            *convert_args,
            str(temp_path),
        ]
        logger.info(f"Conversion command: {' '.join(command)}")
        retry_request(lambda: subprocess.run(command, check=True, text=True, capture_output=True))

    manager.execute(convert)
    return str(model_path)


@pytest.fixture(scope="session", autouse=True)
def session_setup_teardown():
    ov_cache_dir = get_ov_cache_dir()
    ov_cache_converted_dir = get_ov_cache_converted_models_dir()
    logger.info(f"Creating directories: {ov_cache_converted_dir}")
    if not ov_cache_converted_dir.exists():
        ov_cache_converted_dir.mkdir(exist_ok=True, parents=True)

    yield

    if os.environ.get("CLEANUP_CACHE", "false").lower() == "true":
        if ov_cache_dir.exists():
            logger.info(f"Removing temporary directory: {ov_cache_dir}")
            shutil.rmtree(ov_cache_dir)
        else:
            logger.info(f"Skipped temporary directory cleanup because it doesn't exist: {ov_cache_dir}")


def run_wwb(args: list[str], env=None):
    command = ["wwb"] + args
    base_env = {"TRANSFORMERS_VERBOSITY": "debug", "PYTHONIOENCODING": "utf-8", **os.environ}
    if env:
        base_env.update(env)

    return subprocess.check_output(
        command,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        env=base_env,
    )
