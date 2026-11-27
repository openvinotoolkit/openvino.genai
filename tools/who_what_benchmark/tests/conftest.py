import os
import re
import pytest
import logging
import shutil
import subprocess  # nosec B404
import sys
import time

from pathlib import Path
from typing import Any, Callable, Dict

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
    "Qwen3-Reranker-0.6B-seq-cls": {
        "name": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
        "convert_args": ["--trust-remote-code", "--task", "text-classification"],
    },
    "Qwen3-Reranker-0.6B": {
        "name": "Qwen/Qwen3-Reranker-0.6B",
        "convert_args": ["--trust-remote-code", "--task", "text-generation"],
    },
    "tiny-random-Phi3ForCausalLM": {
        "name": "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM",
        "convert_args": ["--trust-remote-code", "--task", "text-generation"],
    },
    "Qwen2.5-0.5B-Instruct": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "cache_name": "Qwen2.5-0.5B-Instruct-v2",
        "convert_args": ["--trust-remote-code"],
    },
    "tiny-random-qwen2vl": {
        "name": "optimum-intel-internal-testing/tiny-random-qwen2vl",
        "convert_args": ["--trust-remote-code", "--task", "image-text-to-text"],
    },
    "tiny-random-qwen3-omni": {
        "name": "optimum-intel-internal-testing/tiny-random-qwen3-omni",
        "convert_args": ["--trust-remote-code", "--task", "image-text-to-text"],
    },
    "tiny-random-gemma4": {
        "name": "optimum-intel-internal-testing/tiny-random-gemma4",
        "convert_args": ["--task", "image-text-to-text"],
    },
    "tiny-random-fun-asr": {
        "name": "optimum-intel-internal-testing/tiny-random-fun-asr",
        "convert_args": ["--weight-format", "fp32"],
    },
    "tiny-random-qwen3-asr": {
        "name": "optimum-intel-internal-testing/tiny-random-qwen3-asr",
        "convert_args": ["--task", "automatic-speech-recognition"],
    },
    "whisper-tiny": {
        "name": "openai/whisper-tiny",
        "convert_args": ["--weight-format", "fp32"],
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
    "tiny-random-mistral3": {
        "name": "optimum-intel-internal-testing/tiny-random-mistral3",
        "convert_args": ["--trust-remote-code", "--task", "image-text-to-text"],
    },
    "Qwen2-VL-2B-Instruct": {
        "name": "Qwen/Qwen2-VL-2B-Instruct",
        "convert_args": ["--trust-remote-code", "--task", "image-text-to-text"],
    },
    "tiny-random-stable-diffusion-xl": {"name": "echarlaix/tiny-random-stable-diffusion-xl", "convert_args": []},
    "stable-diffusion-3-tiny-random": {"name": "yujiepan/stable-diffusion-3-tiny-random", "convert_args": []},
    "tiny-random-flux": {"name": "optimum-intel-internal-testing/tiny-random-flux", "convert_args": []},
    "tiny-random-flux-fill": {"name": "optimum-intel-internal-testing/tiny-random-flux-fill", "convert_args": []},
    "tiny-random-ltx-video": {"name": "optimum-intel-internal-testing/tiny-random-ltx-video", "convert_args": []},
    "tiny-random-ltx2": {"name": "optimum-intel-internal-testing/tiny-random-ltx2", "convert_args": []},
    "tiny-videochat-flash-qwen": {
        "name": "optimum-intel-internal-testing/tiny-videochat-flash-qwen",
        "convert_args": ["--trust-remote-code", "--task", "image-text-to-text"],
    },
    "speecht5_tts": {
        "name": "microsoft/speecht5_tts",
        "convert_args": ["--model-kwargs", '{"vocoder": "microsoft/speecht5_hifigan"}'],
    },
    "Kokoro-82M": {
        "name": "hexgrad/Kokoro-82M",
        "convert_args": ["--trust-remote-code"],
    },
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign": {
        "name": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "convert_args": ["--trust-remote-code", "--weight-format", "fp32"],
    },
    "Qwen3-TTS-12Hz-1.7B-CustomVoice": {
        "name": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "convert_args": ["--trust-remote-code", "--weight-format", "fp32"],
    },
    "Qwen3-TTS-12Hz-0.6B-Base": {
        "name": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "convert_args": ["--trust-remote-code", "--weight-format", "fp32"],
    },
    "tiny-random-qwen3-vl-embedding": {
        "name": "optimum-intel-internal-testing/tiny-random-qwen3-vl-embedding",
        "convert_args": ["--trust-remote-code", "--task", "feature-extraction"],
    },
}


def get_ov_cache_converted_models_dir():
    return get_ov_cache_dir() / "converted_models"


def convert_text_model(
    model_id: str,
    dir_name: str,
    convert_fn: Callable[[str, Path], None],
) -> str:
    models_dir = get_ov_cache_converted_models_dir()
    model_path = Path(models_dir) / f"wwb_{dir_name}"

    manager = AtomicDownloadManager(model_path)

    logger.info(f"Start conversion of: {model_id} -> {dir_name}")
    if manager.is_complete():
        logger.info("Conversion is already completed")
        return str(model_path)

    def convert(temp_path: Path) -> None:
        retry_request(lambda: convert_fn(model_id, temp_path))

    manager.execute(convert)
    return str(model_path)


def convert_model(model_name: str) -> str:
    models_dir = get_ov_cache_converted_models_dir()

    parts = model_name.split("/")
    if len(parts) != 2 or parts[1] not in MODELS:
        raise ValueError(
            f"Invalid model name '{model_name}', potentially it is not registered in MODELS set or the name is incorrect."
        )
    model_id = parts[1]

    convert_args = MODELS[model_id]["convert_args"]
    cache_name = MODELS[model_id].get("cache_name", model_id)
    model_path = Path(models_dir) / f"wwb_{cache_name}"

    manager = AtomicDownloadManager(model_path)

    logger.info(f"Start convertion of: {model_name}")
    if manager.is_complete():
        logger.info("Conversion command is already completed")
        return model_path

    def convert(temp_path: Path) -> None:
        if "--weight-format" not in convert_args:
            convert_args.extend(["--weight-format", "fp16"])
        command = [
            *_export_tool(MODELS[model_id]["name"]),
            "--model",
            MODELS[model_id]["name"],
            *convert_args,
            str(temp_path),
        ]
        logger.info(f"Conversion command: {' '.join(command)}")
        retry_request(lambda: subprocess.run(command, check=True, text=True, encoding="utf-8", capture_output=True))

    try:
        manager.execute(convert)
    except subprocess.CalledProcessError as error:
        logger.exception(f"{error.cmd[0]} returned {error.returncode}. Output:\n{error.stderr}")
        raise

    return str(model_path)


def _export_tool(model: str) -> list[str]:
    """genai-cli for the models it converts natively, optimum-cli for all others (same export options).

    Mirrors export_command of tests/python_tests/utils/export.py, which the WWB tests cannot import. WWB can be
    installed next to an openvino-genai release that has no genai-cli, hence the ImportError fallback.
    """
    try:
        from openvino_genai._cli.main import native_exporter_for_model

        if native_exporter_for_model(model)[1] is not None:
            return [sys.executable, "-m", "openvino_genai._cli", "export"]
    except ImportError:  # openvino_genai built without genai-cli
        pass
    return ["optimum-cli", "export", "openvino"]


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


# wwb output of a run that failed on a Hugging Face Hub rate limit or a network error, which a later run can pass.
# Keep in sync with the --only-rerun pattern of PYTEST_ADDOPTS in the "Tests" step of the genai_tests_wheel jobs.
WWB_TRANSIENT_ERROR = re.compile(
    r"Too Many Requests|\b5\d\d Server Error|Gateway Time-?out|Temporary failure in name resolution"
    r"|Name or service not known|nodename nor servname provided|getaddrinfo failed|Connection reset by peer"
    r"|forcibly closed by the remote host|Server disconnected without sending a response|Read timed out"
    r"|The read operation timed out"
)
# The Hugging Face Hub reports when the rate limit window frees up: "Retry after 45 seconds (0/3000 requests ...)".
WWB_RETRY_AFTER = re.compile(r"Retry after (\d+) seconds")
WWB_DEFAULT_RETRY_DELAY_SECONDS = 30
WWB_MAX_RETRY_DELAY_SECONDS = 120
WWB_MAX_ATTEMPTS = 3
# Caps the total wait of a session, so that a long outage fails the job instead of running into its timeout.
WWB_MAX_SESSION_RETRIES = 2
_wwb_session_retries = 0


def _wwb_retry_delay(output: str) -> int | None:
    """Returns the seconds to wait before rerunning a failed wwb command, or None if the failure is not transient."""
    if not output or not WWB_TRANSIENT_ERROR.search(output):
        return None
    retry_after = [int(seconds) for seconds in WWB_RETRY_AFTER.findall(output)]
    # A few extra seconds so the rerun doesn't start right at the window boundary
    delay = max(retry_after) + 5 if retry_after else WWB_DEFAULT_RETRY_DELAY_SECONDS
    return min(delay, WWB_MAX_RETRY_DELAY_SECONDS)


def run_wwb(args: list[str], env=None):
    global _wwb_session_retries

    command = ["wwb"] + args
    base_env = {"TRANSFORMERS_VERBOSITY": "debug", "PYTHONIOENCODING": "utf-8", **os.environ}
    if env:
        base_env.update(env)

    for attempt in range(1, WWB_MAX_ATTEMPTS + 1):
        try:
            return subprocess.check_output(
                command,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                env=base_env,
            )
        except subprocess.CalledProcessError as error:
            logger.error(f"'{' '.join(map(str, command))}' returned {error.returncode}. Output:\n{error.output}")
            delay = _wwb_retry_delay(error.output)
            if delay is None or attempt == WWB_MAX_ATTEMPTS or _wwb_session_retries >= WWB_MAX_SESSION_RETRIES:
                raise
            _wwb_session_retries += 1
            logger.warning(
                f"wwb failed on a transient error (attempt {attempt}/{WWB_MAX_ATTEMPTS}), retrying in {delay} s"
            )
            time.sleep(delay)
