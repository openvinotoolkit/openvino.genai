import subprocess  # nosec B404
import sys
import pytest
import logging
from test_cli_image import run_wwb


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [
        pytest.param("BAAI/bge-small-en-v1.5", "text-embedding", marks=pytest.mark.xfail(
            sys.platform == 'darwin', reason="Hangs. Ticket 175534", run=False
        )),
        ("Qwen/Qwen3-Embedding-0.6B", "text-embedding"),
    ],
)
def test_embeddings_basic(model_id, model_type, tmp_path):
    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = tmp_path / model_id.replace("/", "_")

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


@pytest.mark.parametrize(
    ("model_id", "model_type", "batch_size"),
    [
        ("Qwen/Qwen3-Embedding-0.6B", "text-embedding", 1),
        ("Qwen/Qwen3-Embedding-0.6B", "text-embedding", 12),
    ],
)
def test_embeddings_with_batch(model_id, model_type, batch_size, tmp_path):
    GT_FILE = tmp_path / f"gt_batch_{batch_size}.csv"
    MODEL_PATH = tmp_path / model_id.replace("/", "_")

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
        "--embeds_batch_size",
        str(batch_size),
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
        "--embeds_batch_size",
        str(batch_size),
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
        "--embeds_batch_size",
        str(batch_size),
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
        "--embeds_batch_size",
        str(batch_size),
    ])
