import sys
import pytest
import logging
import tempfile
from conftest import convert_model, run_wwb
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
tmp_dir = tempfile.mkdtemp()


OV_RERANK_MODELS = {
    ("cross-encoder/ms-marco-TinyBERT-L2-v2", "text-classification"),
    ("Qwen/Qwen3-Reranker-0.6B", "text-generation"),
}


@pytest.mark.parametrize(("model_info"), OV_RERANK_MODELS)
def test_reranking_genai(model_info, tmp_path):
    if sys.platform == 'darwin':
        pytest.xfail("Ticket 175534")
    if sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    GT_FILE = Path(tmp_dir) / "gt.csv"
    model_id = model_info[0]
    MODEL_PATH = convert_model(model_id)

    # test GenAI
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
        "text-reranking",
        "--genai"
    ])

    assert Path(tmp_dir, "reference").exists()


@pytest.mark.parametrize(
    ("model_info"), OV_RERANK_MODELS
)
@pytest.mark.xfail(sys.platform == 'darwin', reason="Hangs. Ticket 175534", run=False)
def test_reranking_optimum(model_info, tmp_path):
    if sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    GT_FILE = Path(tmp_dir) / "gt.csv"
    model_id = model_info[0]
    MODEL_PATH = convert_model(model_id)

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

    # test Optimum
    outpus = run_wwb([
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
        tmp_path,
    ])

    assert (tmp_path / "target").exists()
    assert (tmp_path / "target.csv").exists()
    assert (tmp_path / "metrics_per_question.csv").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert "Metrics for model" in outpus

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
        "text-reranking",
        "--genai"
    ])
