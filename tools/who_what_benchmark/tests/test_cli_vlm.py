import subprocess  # nosec B404
import pytest
import logging
import sys
from test_cli_image import run_wwb
from constants import WWB_CACHE_PATH


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [
        ("katuni4ka/tiny-random-llava", "visual-text"),
    ],
)
def test_vlm_basic(model_id, model_type, tmp_path):
    if sys.platform == 'darwin':
        pytest.xfail("Ticket 173169")
    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = WWB_CACHE_PATH.joinpath(model_id.replace("/", "--"))

    if not MODEL_PATH.exists():
        result = subprocess.run(["optimum-cli", "export",
                                 "openvino", "-m", model_id,
                                 MODEL_PATH, "--task",
                                 "image-text-to-text",
                                 "--trust-remote-code"],
                                capture_output=True,
                                text=True,
                                )
        assert result.returncode == 0

    # Collect reference with HF model
    run_wwb([
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
        "--hf",
    ])

    # test Optimum
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
    ])

    # test GenAI
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
        "--genai",
        "--output",
        WWB_CACHE_PATH,
    ])

    # test w/o models
    run_wwb([
        "--target-data",
        WWB_CACHE_PATH / "target.csv",
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
        "--genai",
    ])
