import subprocess  # nosec B404
import sys
import pytest
import shutil
import logging
import tempfile
from test_cli_image import run_wwb, get_similarity
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
tmp_dir = tempfile.mkdtemp()


def download_model(model_id, task, tmp_path):
    MODEL_PATH = Path(tmp_path, model_id.replace("/", "_"))
    subprocess.run(["optimum-cli", "export", "openvino", "--model", model_id, MODEL_PATH, "--task", task],
                   capture_output=True,
                   text=True)
    return MODEL_PATH


def remove_artifacts(artifacts_path: Path, file_type="outputs"):
    shutil.rmtree(artifacts_path)


@pytest.mark.wwb_rerank
@pytest.mark.parametrize(
    ("model_id", "model_task", "threshold"),
    [
        ("cross-encoder/ms-marco-TinyBERT-L2-v2", "text-classification", 0.6),
        ("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", "text-classification", 0.6),
        ("Qwen/Qwen3-Reranker-0.6B", "text-generation", 0.6),
    ],
)
@pytest.mark.xfail(sys.platform == 'darwin', reason="Hangs. Ticket 175534", run=False)
def test_reranking_optimum(model_id, model_task, threshold, tmp_path):
    GT_FILE = Path(tmp_dir) / "gt.csv"
    MODEL_PATH = download_model(model_id, model_task, tmp_path)

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
        "text-reranking",
        "--hf",
    ])

    assert GT_FILE.exists()
    assert Path(tmp_dir, "reference").exists()

    outputs_path = tmp_path / "optimum"
    # test Optimum
    outputs_optimum = run_wwb([
        "--target-model",
        MODEL_PATH,
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        "text-reranking",
        "--output",
        outputs_path,
    ])

    assert (outputs_path / "target").exists()
    assert (outputs_path / "target.csv").exists()
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert "Metrics for model" in outputs_optimum

    similarity = get_similarity(outputs_optimum)
    assert similarity >= threshold

    remove_artifacts(outputs_path)

    outputs_path = tmp_path / "genai"
    # test GenAI
    outputs_genai = run_wwb([
        "--target-model",
        MODEL_PATH,
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        "text-reranking",
        "--genai",
        "--output",
        outputs_path,
    ])
    assert (outputs_path / "target").exists()
    assert (outputs_path / "target.csv").exists()
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert "Metrics for model" in outputs_genai

    similarity = get_similarity(outputs_genai)
    assert similarity >= threshold

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
        "text-reranking",
        "--genai"
    ])

    remove_artifacts(outputs_path)
    remove_artifacts(MODEL_PATH, "model")
