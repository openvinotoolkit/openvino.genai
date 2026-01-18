import itertools
import subprocess  # nosec B404
import os
import sys
import shutil
import pytest
import logging
import tempfile
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CACHE = tempfile.mkdtemp()
OV_IMAGE_MODELS = [
    "optimum-intel-internal-testing/tiny-random-stable-diffusion-xl",
    "optimum-intel-internal-testing/stable-diffusion-3-tiny-random",
    "optimum-intel-internal-testing/tiny-random-flux",
    "optimum-intel-internal-testing/tiny-random-flux-fill",
]


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _log(message: str) -> None:
    logger.info(f"[{_ts()}] [wwb] {message}")


@contextmanager
def _stage(name: str):
    _log(f"START {name}")
    start = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - start
        _log(f"END   {name} dt={dt:.3f}s")


def _require_executable(exe: str, *, context: str) -> None:
    path = shutil.which(exe)
    if path is None:
        pytest.skip(f"Missing required executable '{exe}' for {context}. Ensure it is on PATH.")
    _log(f"Using {exe} at: {path}")


def _truncate(s: str, limit: int = 4000) -> str:
    if s is None:
        return ""
    return s if len(s) <= limit else (s[:limit] + "...<truncated>")


def run_wwb(args, env=None):
    _require_executable("wwb", context="WWB CLI tests")
    command = ["wwb"] + args
    base_env = {"TRANSFORMERS_VERBOSITY": "debug", "PYTHONIOENCODING": "utf-8", **os.environ}
    if env:
        base_env.update(env)
    _log("Command: " + " ".join(map(str, command)))
    try:
        with _stage("wwb_run"):
            return subprocess.check_output(
                command,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                env=base_env,
            )
    except subprocess.CalledProcessError as error:
        logger.error(
            f"'{' '.join(map(str, command))}' returned {error.returncode}. Output:\n"
            f"{error.output}"
        )
        raise


def setup_module():
    _require_executable("optimum-cli", context="OpenVINO model export in WWB CLI tests")
    _log(f"MODEL_CACHE={MODEL_CACHE}")
    for model_id in OV_IMAGE_MODELS:
        MODEL_PATH = os.path.join(MODEL_CACHE, model_id.replace("/", "_"))
        _log(f"Export OpenVINO model: model_id={model_id} -> {MODEL_PATH}")
        with _stage("optimum_cli_export"):
            result = subprocess.run(
                ["optimum-cli", "export", "openvino", "--model", model_id, MODEL_PATH],
                capture_output=True,
                text=True,
            )
        assert result.returncode == 0, (
            f"optimum-cli export failed for model_id={model_id} rc={result.returncode}\n"
            f"stdout:\n{_truncate(result.stdout)}\n"
            f"stderr:\n{_truncate(result.stderr)}"
        )


def teardown_module():
    logger.info("Remove models")
    shutil.rmtree(MODEL_CACHE)


def get_similarity(output: str) -> float:
    METRIC_PATTERN = "INFO:whowhatbench.wwb:   similarity"
    substr = output[output.find(METRIC_PATTERN) + len(METRIC_PATTERN) + 1:]
    float_pattern = r"[-+]?\d*\.\d+"
    matches = re.findall(float_pattern, substr)
    return float(matches[-1])


@pytest.mark.parametrize(
    ("model_id", "model_type", "backend"),
    [
        ("hf-internal-testing/tiny-stable-diffusion-torch", "image-to-image", "hf"),
        ("hf-internal-testing/tiny-stable-diffusion-xl-pipe", "image-to-image", "hf"),
        ("hf-internal-testing/tiny-stable-diffusion-torch", "text-to-image", "hf"),
        ("hf-internal-testing/tiny-stable-diffusion-torch", "text-to-image", "openvino"),
        ("hf-internal-testing/tiny-stable-diffusion-xl-pipe", "text-to-image", "hf"),
        ("hf-internal-testing/tiny-stable-diffusion-torch", "image-inpainting", "hf"),
        ("hf-internal-testing/tiny-stable-diffusion-xl-pipe", "image-inpainting", "hf"),
    ],
)
def test_image_model_types(model_id, model_type, backend, tmp_path):
    if 'tiny-stable-diffusion-torch' in model_id and sys.platform == 'darwin':
        pytest.xfail("Ticket 173169")
    if (model_type == "image-to-image" or model_type == "image-inpainting") and sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    wwb_args = [
        "--base-model",
        model_id,
        "--target-model",
        model_id,
        "--num-samples",
        "1",
        "--gt-data",
        tmp_path / "test_sd.csv",
        "--device",
        "CPU",
        "--model-type",
        model_type,
        "--num-inference-steps",
        "2",
    ]
    if backend == "hf":
        wwb_args.append("--hf")
    elif backend == "genai":
        wwb_args.append("--genai")

    output = run_wwb(wwb_args)

    assert "Metrics for model" in output
    similarity = get_similarity(output)
    assert similarity >= 0.98


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    list(itertools.product(OV_IMAGE_MODELS,
                           ["image-to-image",
                            "text-to-image",
                            "image-inpainting"
                            ])),
)
def test_image_model_genai(model_id, model_type, tmp_path):
    if ("flux-fill" in model_id) and (model_type != "image-inpainting"):
        pytest.skip(reason="FLUX-Fill is supported as inpainting only")
    if model_id == "optimum-intel-internal-testing/tiny-random-flux" and model_type == "image-to-image":
        pytest.xfail("Randomly wwb died with <Signals.SIGABRT: 6>. Ticket 170878")
    if (model_type == "image-to-image" or model_type == "image-inpainting") and sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    mac_arm64_skip = any(substring in model_id for substring in ('stable-diffusion-xl',
                                                                 'tiny-random-stable-diffusion',
                                                                 'stable-diffusion-3',
                                                                 'tiny-random-flux'))

    if mac_arm64_skip and sys.platform == 'darwin':
        pytest.xfail("Ticket 173169")

    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = os.path.join(MODEL_CACHE, model_id.replace("/", "_"))

    run_wwb([
        "--base-model",
        model_id,
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
        "--num-inference-steps",
        "2",
    ])
    assert GT_FILE.exists()
    assert (tmp_path / "reference").exists()

    output = run_wwb([
        "--target-model",
        MODEL_PATH,
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
        "--genai",
        "--num-inference-steps",
        "2",
    ])

    assert "Metrics for model" in output
    similarity = get_similarity(output)
    assert similarity >= 0.97751  # Ticket 166496
    assert (tmp_path / "target").exists()

    run_wwb([
        "--target-model",
        MODEL_PATH,
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
        "--output",
        tmp_path,
        "--genai",
        "--num-inference-steps",
        "2",
    ])
    assert (tmp_path / "target").exists()
    assert (tmp_path / "target.csv").exists()

    # test w/o models
    run_wwb([
        "--target-data",
        tmp_path / "target.csv",
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
        "--num-inference-steps",
        "2",
    ])


@pytest.mark.parametrize(
    ("model_id", "model_type", "backend"),
    [
        ("hf-internal-testing/tiny-stable-diffusion-torch", "text-to-image", "hf"),
    ],
)
def test_image_custom_dataset(model_id, model_type, backend, tmp_path):
    if sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    GT_FILE = tmp_path / "test_sd.csv"
    wwb_args = [
        "--base-model",
        model_id,
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
        "--dataset",
        "google-research-datasets/conceptual_captions",
        "--dataset-field",
        "caption",
        "--num-inference-steps",
        "2",
    ]
    if backend == "hf":
        wwb_args.append("--hf")
    elif backend == "genai":
        wwb_args.append("--genai")

    run_wwb(wwb_args)

    assert os.path.exists(GT_FILE)
