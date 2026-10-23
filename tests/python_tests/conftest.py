# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
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
    parser.addoption("--gguf", action="store_true", help="Include frontend GGUF models in shared LLM tests")


def pytest_configure(config: pytest.Config):
    pytest.selected_model_ids = config.getoption("--model_ids", default=None)
    pytest.include_gguf = config.getoption("--gguf")


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


@pytest.fixture(scope="module", autouse=True)
def run_gc_after_test():
    """
    Fixture to run garbage collection after each test module.
    This is a workaround to minimize memory consumption during tests 
    and allow the use of less powerful CI runners.
    """
    yield
    gc.collect()


def pytest_collection_modifyitems(items):
    from openvino_genai import GenerationConfig
    from data.models import GGUFModelDescription
    from utils.ov_genai_pipelines import PipelineType, is_generation_available

    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue
        model = callspec.params.get("llm_model")
        if not isinstance(model, GGUFModelDescription):
            continue
        item.add_marker(
            pytest.mark.skipif(
                sys.platform in {"darwin", "win32"},
                reason="GGUF frontend integration: CVS-168882 (macOS), CVS-174065 (Windows)",
            )
        )
        pipeline_type = callspec.params.get("pipeline_type", PipelineType.AUTO)
        generation_config_dict = callspec.params.get("generation_config_dict")
        if generation_config_dict is not None and not is_generation_available(
            GenerationConfig(**generation_config_dict), pipeline_type
        ):
            item.add_marker(
                pytest.mark.skip(reason="Beam search is unsupported by speculative and prompt lookup decoding")
            )
            continue
        test_name = item.originalname
        reason = None
        raises = AssertionError
        run = True
        if pipeline_type == PipelineType.SPECULATIVE_DECODING:
            reason = "draft_model(path) expects an IR directory and cannot load a GGUF file"
            raises = RuntimeError
        elif (
            test_name == "test_perf_metrics_with_structured_output"
            and model.gguf_repo != "mradermacher/tiny-random-qwen3-GGUF"
        ):
            reason = "GGUF SentencePiece tokenizer does not expose the full vocabulary required by structured output"
            raises = RuntimeError
        elif model.gguf_repo == "aladar/llama-2-tiny-random-GGUF":
            if test_name in {"test_streamer_kwargs_batch_throws", "test_operator_with_streamer_kwargs_batch_throws"}:
                reason = "Llama GGUF empty prompt with beam search and a streamer crashes the native pipeline"
                run = False
            elif test_name == "test_batch_string_inputs" and any(
                prompt.endswith(" ") for prompt in callspec.params["prompts"]
            ):
                reason = "GGUF Llama tokenizer drops trailing whitespace preserved by the HF reference"
        elif model.gguf_repo == "mradermacher/tiny-random-qwen3-GGUF" and test_name == "test_batch_string_inputs":
            if callspec.params["prompts"] in [
                ["table is made", "They sky is blue because", "Difference between Jupiter and Mars is that"],
                ["Alan Turing was a", "return 0", "你好！ 你好嗎？"],
                ["table is made", "table is made [force left pad tokens]"],
            ]:
                reason = "Tiny Qwen3 GGUF batched generation differs from the exact-weight Optimum reference"
        elif model.gguf_repo == "mradermacher/tiny-random-Phi3ForCausalLM-GGUF":
            if test_name in {
                "test_string_inputs",
                "test_batch_string_inputs",
                "test_chat_scenario",
                "test_chat_scenario_callback_cancel",
            }:
                reason = "Tiny Phi-3 GGUF text/chat generation differs from the exact-weight Optimum reference"
        if reason:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=True, raises=raises, run=run))
