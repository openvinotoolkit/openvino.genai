import subprocess  # nosec B404
import sys
import pytest
import shutil
import logging
from test_cli_image import run_wwb, get_similarity


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def remove_artifacts(artifacts_path, file_type="outputs"):
    logger.info(f"Remove {file_type}")
    shutil.rmtree(artifacts_path)


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
    SIMILARITY_THRESHOLD = 0.99

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

    outputs_path = tmp_path / "optimum"
    # test Optimum
    outputs = run_wwb([
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
        outputs_path,
    ])

    assert (outputs_path / "target").exists()
    assert (outputs_path / "target.csv").exists()
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert "Metrics for model" in outputs

    similarity = get_similarity(outputs)
    assert similarity >= SIMILARITY_THRESHOLD

    remove_artifacts(outputs_path.as_posix())

    outputs_path = tmp_path / "genai"
    # test GenAI
    outputs = run_wwb([
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
        outputs_path,
    ])

    assert (outputs_path / "target").exists()
    assert (outputs_path / "target.csv").exists()
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert "Metrics for model" in outputs

    similarity = get_similarity(outputs)
    assert similarity >= SIMILARITY_THRESHOLD

    # test w/o models
    run_wwb([
        "--target-data",
        outputs_path / "target.csv",
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

    remove_artifacts(outputs_path.as_posix())
    remove_artifacts(MODEL_PATH.as_posix(), "model")
