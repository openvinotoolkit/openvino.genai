import itertools
import subprocess  # nosec B404
import os
import sys
import shutil
import pytest
import logging
import tempfile
import re


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CACHE = tempfile.mkdtemp()
OV_IMAGE_MODELS = ["echarlaix/tiny-random-stable-diffusion-xl",
                   "yujiepan/stable-diffusion-3-tiny-random",
                   "katuni4ka/tiny-random-flux",
                   "katuni4ka/tiny-random-flux-fill"]


def run_wwb(args):
    command = ["wwb"] + args
    try:
        return subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            env={"TRANSFORMERS_VERBOSITY": "debug", "PYTHONIOENCODING": "utf-8", **os.environ},
        )
    except subprocess.CalledProcessError as error:
        logger.error(
            f"'{' '.join(map(str, command))}' returned {error.returncode}. Output:\n"
            f"{error.output}"
        )
        raise


def setup_module():
    for model_id in OV_IMAGE_MODELS:
        MODEL_PATH = os.path.join(MODEL_CACHE, model_id.replace("/", "--"))
        subprocess.run(["optimum-cli", "export", "openvino", "--model", model_id, MODEL_PATH], capture_output=True, text=True)


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
    if model_type == "image-inpainting":
        pytest.xfail("Segfault. Ticket 170877")

    mac_arm64_skip = any(substring in model_id for substring in ('stable-diffusion-xl',
                                                                 'tiny-random-stable-diffusion',
                                                                 'stable-diffusion-3',
                                                                 'tiny-random-flux'))

    if mac_arm64_skip and sys.platform == 'darwin':
        pytest.xfail("Ticket 173169")

    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = os.path.join(MODEL_CACHE, model_id.replace("/", "--"))

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
