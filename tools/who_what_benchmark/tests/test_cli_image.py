import subprocess  # nosec B404
import os
import shutil
import pytest
import logging
import tempfile


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_wwb(args):
    logger.info(" ".join(["TRANSFOREMRS_VERBOSITY=debug wwb"] + args))
    result = subprocess.run(["wwb"] + args, capture_output=True, text=True)
    logger.info(result)
    print(" ".join(["TRANSFOREMRS_VERBOSITY=debug wwb"] + args))
    return result


@pytest.mark.parametrize(
    ("model_id", "model_type", "backend"),
    [
        ("hf-internal-testing/tiny-stable-diffusion-torch", "text-to-image", "hf"),
        ("hf-internal-testing/tiny-stable-diffusion-torch", "text-to-image", "openvino"),
        ("hf-internal-testing/tiny-stable-diffusion-xl-pipe", "text-to-image", "hf"),
    ],
)
def test_image_model_types(model_id, model_type, backend):
    GT_FILE = "test_sd.json"
    wwb_args = [
        "--base-model",
        model_id,
        "--target-model",
        model_id,
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
    ]
    if backend == "hf":
        wwb_args.append("--hf")
    elif backend == "genai":
        wwb_args.append("--genai")

    result = run_wwb(wwb_args)
    print(result.stderr, result.stdout)

    try:
        os.remove(GT_FILE)
    except OSError:
        pass
    shutil.rmtree("reference", ignore_errors=True)
    shutil.rmtree("target", ignore_errors=True)

    assert result.returncode == 0
    assert "Metrics for model" in result.stderr
    similarity = float(str(result.stderr).split(" ")[-1])
    assert similarity >= 0.98


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [
        ("echarlaix/tiny-random-stable-diffusion-xl", "text-to-image"),
    ],
)
def test_image_model_genai(model_id, model_type):
    GT_FILE = "test_sd.json"
    MODEL_PATH = tempfile.TemporaryDirectory().name

    result = subprocess.run(["optimum-cli", "export",
                             "openvino", "-m", model_id,
                             MODEL_PATH], capture_output=True, text=True)
    assert result.returncode == 0

    wwb_args = [
        "--base-model",
        MODEL_PATH,
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
        "--image-size",
        "256",
    ]
    result = run_wwb(wwb_args)
    assert result.returncode == 0

    wwb_args = [
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
        "--image-size",
        "256",
    ]
    result = run_wwb(wwb_args)

    try:
        os.remove(GT_FILE)
    except OSError:
        pass
    shutil.rmtree("reference", ignore_errors=True)
    shutil.rmtree("target", ignore_errors=True)
    shutil.rmtree(MODEL_PATH, ignore_errors=True)

    assert result.returncode == 0
    assert "Metrics for model" in result.stderr
    similarity = float(str(result.stderr).split(" ")[-1])
    assert similarity >= 0.98


@pytest.mark.parametrize(
    ("model_id", "model_type", "backend"),
    [
        ("hf-internal-testing/tiny-stable-diffusion-torch", "text-to-image", "hf"),
    ],
)
def test_image_custom_dataset(model_id, model_type, backend):
    GT_FILE = "test_sd.json"
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
    ]
    if backend == "hf":
        wwb_args.append("--hf")
    elif backend == "genai":
        wwb_args.append("--genai")

    result = run_wwb(wwb_args)

    assert os.path.exists(GT_FILE)

    try:
        os.remove(GT_FILE)
    except OSError:
        pass
    shutil.rmtree("reference", ignore_errors=True)

    assert result.returncode == 0
