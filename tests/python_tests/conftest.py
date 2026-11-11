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
from utils.export import export_command, export_with_genai_cli, native_genai_cli_model_type

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PROMPTS: list[str] = [
    "What is in the image?",
    "What is special about this image?",
    "Describe the image"
]

ATTENTION_BACKEND: list[str] = ["PA", "SDPA"]

TEST_IMAGE_URLS = {
    'cat': 'https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11',
    'car': 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg',
    'handwritten': 'https://github.com/user-attachments/assets/8c9ae017-7837-4abc-ae92-c1054c9ec350'
}


def _get_ov_model(model_id: str) -> str:
    import utils.patch_pyav_for_servercore as patch_pyav_for_servercore

    patch_pyav_for_servercore.install_av_stub_module_for_windows()

    import openvino_tokenizers
    import openvino
    import transformers
    from optimum.intel.openvino import OVModelForVisualCausalLM
    from optimum.utils.import_utils import is_transformers_version
    from huggingface_hub import snapshot_download

    ov_cache_converted_dir = get_ov_cache_converted_models_dir()
    dir_name = str(model_id).replace(os.sep, "_")
    model_dir = ov_cache_converted_dir / dir_name
    manager = AtomicDownloadManager(model_dir)

    if manager.is_complete() or (model_dir / "openvino_language_model.xml").exists():
        return model_dir

    def convert_to_temp(temp_dir: Path) -> None:
        model_cached = retry_request(lambda: snapshot_download(model_id))
        if native_genai_cli_model_type(model_cached) is not None:
            export_with_genai_cli(model_cached, temp_dir, ["--trust-remote-code"])
            return
        align_with_optimum_cli = {"padding_side": "left", "truncation_side": "left"}
        processor = retry_request(
            lambda: transformers.AutoProcessor.from_pretrained(
                model_cached,
                trust_remote_code=True,
                **align_with_optimum_cli,
            )
        )
        model = retry_request(
            lambda: OVModelForVisualCausalLM.from_pretrained(
                model_cached,
                compile=False,
                device="CPU",
                export=True,
                load_in_8bit=False,
                trust_remote_code=model_id in {
                    "optimum-intel-internal-testing/tiny-random-minicpmv-2_6",
                    "optimum-intel-internal-testing/tiny-random-minicpm-v-4_5",
                    "optimum-intel-internal-testing/tiny-random-internvl2",
                    "optimum-intel-internal-testing/tiny-random-phi3-vision",
                    "optimum-intel-internal-testing/tiny-random-phi-4-multimodal",
                    "qnguyen3/nanoLLaVA",
                    "optimum-intel-internal-testing/tiny-random-MiniCPM-o-2_6",
                    "optimum-intel-internal-testing/tiny-videochat-flash-qwen",
                },
            )
        )
        if model.config.model_type == "llava-qwen2" or "videochat-flash-qwen" in model_id.lower():
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_cached, trust_remote_code=True)
        # For tiny-random-internvl2 processor is actually tokenizer
        elif isinstance(processor, transformers.Qwen2TokenizerFast):
            tokenizer = processor
            processor = transformers.AutoImageProcessor.from_pretrained(model_cached, trust_remote_code=True)
        else:
            tokenizer = processor.tokenizer
            if tokenizer.chat_template is None:
                tokenizer.chat_template = processor.chat_template
        tokenizer.save_pretrained(temp_dir)
        ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(
            tokenizer, with_detokenizer=True
        )
        openvino.save_model(ov_tokenizer, temp_dir / "openvino_tokenizer.xml")
        openvino.save_model(ov_detokenizer, temp_dir / "openvino_detokenizer.xml")

        if tokenizer.chat_template is not None and model.config.model_type == "phi3_v":
            # It seems that tiny-random-phi3-vision is saved incorrectly. That line works this around.
            processor.chat_template = tokenizer.chat_template

        if model_id not in [
            "optimum-intel-internal-testing/tiny-random-gemma4",
            "optimum-intel-internal-testing/tiny-random-gemma4-unified-it",
        ]:
            if (
                isinstance(processor, getattr(transformers, "Gemma4Processor", type(None)))
                or model.config.model_type == "qwen3_5"
                or is_transformers_version(">=", "5.0")
            ):
                # Setting to None is insufficient because some processors still serialize
                # audio_tokenizer and access its name_or_path attribute.
                processor.__dict__.pop("audio_tokenizer", None)
            else:
                processor.audio_tokenizer = None

        # DeepSeek-OCR-2 tiny-random currently may miss preprocessor_config.json
        # So override processor with AutoImageProcessor in this case.
        if model_id == "optimum-intel-internal-testing/tiny-random-deepseek-ocr-2":
            try:
                processor = transformers.AutoImageProcessor.from_pretrained(
                    model_cached,
                    trust_remote_code=True,
                )
            except (OSError, ValueError):
                pass

        processor.save_pretrained(temp_dir)
        if model.config.model_type == "mistral3":
            # Align with optimum-cli: HF repo nests image processor settings in processor_config.json only.
            processor.image_processor.save_pretrained(temp_dir)
        model.save_pretrained(temp_dir)

    manager.execute(convert_to_temp)
    return model_dir


def retry(func, exception_type=AssertionError):
    __tracebackhide__ = True
    max_retries = 20
    for idx in range(max_retries):
        try:
            return func()
        except exception_type:
            if idx == max_retries - 1:
                raise


def download_image(link: str):
    import PIL.Image
    import requests

    return PIL.Image.open(requests.get(link, stream=True).raw).convert("RGB")


def from_cache_or_download(pytestconfig: pytest.Config, link: str, file_name: str):
    import PIL.Image

    def implementation():
        try:
            image_path = pytestconfig.cache.mkdir("images") / file_name
        except AttributeError:
            # Cache is disabled with -p no:cacheprovider
            return download_image(link)
        if image_path.exists():
            image: PIL.Image = PIL.Image.open(image_path)
        else:
            image = download_image(link)
            image.save(image_path)
        return image
    return retry(implementation, PIL.UnidentifiedImageError)


@pytest.fixture(scope="module")
def cat_image(pytestconfig: pytest.Config):
    return from_cache_or_download(pytestconfig, TEST_IMAGE_URLS['cat'], "cat.jpg")


@pytest.fixture(scope="module")
def cat_tensor(cat_image):
    import openvino

    return openvino.Tensor(cat_image)


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

# Hub doesn't provide pipeline_tag for these models, so the export task can't be inferred automatically.
IMAGE_GEN_MODEL_EXPORT_TASKS = {
    "tiny-random-qwen-image-2.1": "text-to-image",
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
        options = ["--trust-remote-code", "--weight-format", "fp16"]
        export_task = IMAGE_GEN_MODEL_EXPORT_TASKS.get(model_id)
        if export_task is not None:
            options += ["--task", export_task]
        command = export_command(model_name, temp_path, options)
        logger.info(f"Conversion command: {' '.join(command)}")
        retry_request(lambda: subprocess.run(command, check=True, encoding="utf-8", text=True, capture_output=True))

    try:
        manager.execute(convert_model)
    except subprocess.CalledProcessError as error:
        logger.exception(
            f"{error.cmd[0]} returned {error.returncode}. Stdout:\n{error.stdout}\nStderr:\n{error.stderr}"
        )
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
