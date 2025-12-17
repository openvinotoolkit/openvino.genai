import sys
import pytest
import shutil
import logging
from pathlib import Path
from test_cli_image import get_similarity
from conftest import convert_model, run_wwb
from profile_utils import _log, _stage


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def remove_artifacts(artifacts_path: Path):
    shutil.rmtree(artifacts_path)


@pytest.mark.wwb_rerank
@pytest.mark.parametrize(
    ("model_id", "threshold"),
    [
        ("cross-encoder/ms-marco-TinyBERT-L2-v2", 0.99),
        ("Qwen/Qwen3-Reranker-0.6B", 0.99),
        ("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", 0.99),
    ],
)
@pytest.mark.xfail(sys.platform == "darwin", reason="Hangs. Ticket 175534", run=False)
@pytest.mark.xfail(sys.platform == "win32", reason="Ticket 178790", run=False)
def test_reranking_optimum(model_id, threshold, tmp_path):
    GT_FILE = Path(tmp_path) / "gt.csv"
    MODEL_PATH = convert_model(model_id)

    # Collect reference with HF model
    with _stage("run_wwb_hf_reference"):
        run_wwb(
        [
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
        ]
    )

    assert GT_FILE.exists()
    assert Path(tmp_path, "reference").exists()

    outputs_path = tmp_path / "optimum"
    # test Optimum
    with _stage("run_wwb_optimum"):
        outputs_optimum = run_wwb(
        [
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
        ]
    )

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
    with _stage("run_wwb_genai"):
        outputs_genai = run_wwb(
        [
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
        ]
    )
    assert (outputs_path / "target").exists()
    assert (outputs_path / "target.csv").exists()
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert "Metrics for model" in outputs_genai

    similarity = get_similarity(outputs_genai)
    assert similarity >= threshold

    # test w/o models
    with _stage("run_wwb_metrics_without_models"):
        run_wwb(
        [
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
            "--genai",
        ]
    )

    remove_artifacts(outputs_path)
