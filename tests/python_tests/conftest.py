# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import gc
import pytest
import shutil
import subprocess  # nosec B404
import logging
from pathlib import Path

from utils.constants import (
    get_ov_cache_dir,
    get_ov_cache_downloaded_models_dir,
    get_ov_cache_converted_models_dir,
)
from utils.atomic_download import AtomicDownloadManager
from utils.network import retry_request

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


IMAGE_GEN_MODELS = {
    "tiny-random-latent-consistency": "echarlaix/tiny-random-latent-consistency",
    "tiny-random-flux": "optimum-intel-internal-testing/tiny-random-flux",
    "tiny-random-sdxl": "echarlaix/tiny-random-stable-diffusion-xl",
    "tiny-random-sd3": "optimum-intel-internal-testing/stable-diffusion-3-tiny-random",
    "tiny-random-flux.2-klein": "optimum-intel-internal-testing/tiny-random-flux.2-klein",
    "tiny-random-qwenimage": "optimum-intel-internal-testing/tiny-random-qwen-image",
    "tiny-random-qwen-image-2.1": "optimum-intel-internal-testing/tiny-random-qwen-image-2.1",
    "tiny-random-z-image-turbo": "snake7gun/tiny-random-z-image-turbo",
}

DEFAULT_IMAGE_GEN_MODEL_ID = "tiny-random-latent-consistency"


@pytest.fixture(scope="module")
def image_generation_model(request):
    model_id = getattr(request, "param", DEFAULT_IMAGE_GEN_MODEL_ID)
    model_name = IMAGE_GEN_MODELS[model_id]
    models_dir = get_ov_cache_converted_models_dir()
    model_path = Path(models_dir) / model_id / model_name

    manager = AtomicDownloadManager(model_path)

    def convert_model(temp_path: Path) -> None:
        command = [
            "optimum-cli",
            "export",
            "openvino",
            "--model",
            model_name,
            "--trust-remote-code",
            "--weight-format",
            "fp16",
            str(temp_path),
        ]
        logger.info(f"Conversion command: {' '.join(command)}")
        retry_request(lambda: subprocess.run(command, check=True, encoding="utf-8", text=True, capture_output=True))

    try:
        manager.execute(convert_model)
    except subprocess.CalledProcessError as error:
        logger.exception(f"optimum-cli returned {error.returncode}. Stdout:\n{error.stdout}\nStderr:\n{error.stderr}")
        raise

    return str(model_path)


OMNI_MODEL_ID = "optimum-intel-internal-testing/tiny-random-qwen3-omni"


def _export_tiny_omni_model(target_dir: Path) -> None:
    """Export the tiny Qwen3-Omni checkpoint to OpenVINO IR under ``target_dir``."""
    # Imported here so that sessions without Omni tests do not load optimum and transformers.
    import openvino as ov
    import openvino_tokenizers
    import transformers
    from huggingface_hub import snapshot_download
    from optimum.intel.openvino import OVModelForVisualCausalLM

    # Only the download needs retrying; both from_pretrained() calls below read the local path it returns.
    model_cached = retry_request(lambda: snapshot_download(OMNI_MODEL_ID))
    align_with_optimum_cli = {"padding_side": "left", "truncation_side": "left"}
    processor = transformers.AutoProcessor.from_pretrained(
        model_cached,
        trust_remote_code=True,
        **align_with_optimum_cli,
    )
    model = OVModelForVisualCausalLM.from_pretrained(
        model_cached, compile=False, device="CPU", export=True, load_in_8bit=False
    )

    tokenizer = processor.tokenizer
    tokenizer.save_pretrained(target_dir)
    ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(tokenizer, with_detokenizer=True)
    ov.save_model(ov_tokenizer, target_dir / "openvino_tokenizer.xml")
    ov.save_model(ov_detokenizer, target_dir / "openvino_detokenizer.xml")

    processor.save_pretrained(target_dir)
    model.save_pretrained(target_dir)


@pytest.fixture(scope="module")
def omni_model_path() -> Path:
    """Path to an exported tiny Qwen3-Omni model, or skip when the pinned deps cannot produce one.

    Only the known transformers 5.0.x config bug is turned into a skip — any other export failure
    propagates so a real regression cannot hide behind a skipped test.
    """
    model_dir = get_ov_cache_converted_models_dir() / OMNI_MODEL_ID.replace("/", "_")
    manager = AtomicDownloadManager(model_dir)

    if not manager.is_complete() and not (model_dir / "openvino_language_model.xml").exists():
        try:
            manager.execute(_export_tiny_omni_model)
        except AttributeError as error:
            from optimum.utils.import_utils import is_transformers_version

            # Transformers 5.0 reads an uninitialized use_sliding_window in this config; 5.1 fixed it.
            message = str(error)
            if not (
                is_transformers_version(">=", "5.0")
                and is_transformers_version("<", "5.1")
                and "Qwen3OmniMoeTalkerCodePredictorConfig" in message
                and "use_sliding_window" in message
            ):
                raise
            pytest.skip(
                f"Cannot export {OMNI_MODEL_ID}: hit the known transformers 5.0.x config bug, "
                f"fixed in 5.1.0. AttributeError: {message}"
            )

    return model_dir


@pytest.fixture(scope="module", autouse=True)
def run_gc_after_test():
    """
    Fixture to run garbage collection after each test module.
    This is a workaround to minimize memory consumption during tests 
    and allow the use of less powerful CI runners.
    """
    yield
    gc.collect()
