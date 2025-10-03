import subprocess  # nosec B404
import pytest
import logging
from test_cli_image import run_wwb


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [
        ("BAAI/bge-small-en-v1.5", "text-embedding"),
        ("Qwen/Qwen3-Embedding-0.6B", "text-embedding"),
    ],
)
def test_embeddings_basic(model_id, model_type, tmp_path):
    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = tmp_path / model_id.replace("/", "--")

    result = subprocess.run(["optimum-cli", "export",
                             "openvino", "-m", model_id,
                             MODEL_PATH, "--task",
                             "feature-extraction",
                             "--trust-remote-code"],
                            capture_output=True,
                            text=True,
                            )
    assert result.returncode == 0

    # Collect reference with HF model
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
        tmp_path,
    ])

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
        "--genai",
    ])
