import itertools
import subprocess  # nosec B404
import os
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
                   "katuni4ka/tiny-random-flux"]


def run_wwb(args):
    logger.info(" ".join(["TRANSFOREMRS_VERBOSITY=debug wwb"] + args))
    result = subprocess.run(["wwb"] + args, capture_output=True, text=True)
    logger.info(result)
    return result


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
def test_image_model_types(model_id, model_type, backend):
    GT_FILE = "test_sd.csv"
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
        "--num-inference-steps",
        "2",
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
    similarity = get_similarity(str(result.stderr))
    assert similarity >= 0.98


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    list(itertools.product(OV_IMAGE_MODELS,
                           ["image-to-image",
                            "text-to-image",
                            "image-inpainting"
                            ])),
)
def test_image_model_genai(model_id, model_type):
    if ("stable-diffusion-3" in model_id) and model_type != "text-to-image":
        pytest.skip(reason="SD3 is supported as text to image only")

    with tempfile.TemporaryDirectory() as temp_dir:
        GT_FILE = os.path.join(temp_dir, "gt.csv")
        MODEL_PATH = os.path.join(MODEL_CACHE, model_id.replace("/", "--"))

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
            "--num-inference-steps",
            "2",
        ]
        result = run_wwb(wwb_args)
        assert result.returncode == 0
        assert os.path.exists(GT_FILE)
        assert os.path.exists(os.path.join(temp_dir, "reference"))

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
            "--num-inference-steps",
            "2",
        ]
        result = run_wwb(wwb_args)

        assert result.returncode == 0
        assert "Metrics for model" in result.stderr
        similarity = get_similarity(str(result.stderr))
        assert similarity >= 0.98
        assert os.path.exists(os.path.join(temp_dir, "target"))

        output_dir = tempfile.TemporaryDirectory().name
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
            "--output",
            output_dir,
            "--genai",
            "--num-inference-steps",
            "2",
        ]
        result = run_wwb(wwb_args)
        assert result.returncode == 0
        assert os.path.exists(os.path.join(output_dir, "target"))
        assert os.path.exists(os.path.join(output_dir, "target.csv"))

        # test w/o models
        wwb_args = [
            "--target-data",
            os.path.join(output_dir, "target.csv"),
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
        ]
        result = run_wwb(wwb_args)
        assert result.returncode == 0

        shutil.rmtree("reference", ignore_errors=True)
        shutil.rmtree("target", ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)


@pytest.mark.parametrize(
    ("model_id", "model_type", "backend"),
    [
        ("hf-internal-testing/tiny-stable-diffusion-torch", "text-to-image", "hf"),
    ],
)
def test_image_custom_dataset(model_id, model_type, backend):
    GT_FILE = "test_sd.csv"
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

    result = run_wwb(wwb_args)

    assert os.path.exists(GT_FILE)

    try:
        os.remove(GT_FILE)
    except OSError:
        pass
    shutil.rmtree("reference", ignore_errors=True)

    assert result.returncode == 0
