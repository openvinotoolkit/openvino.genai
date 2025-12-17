import subprocess  # nosec B404
import sys
import pytest
import shutil
import logging
import tempfile
import time
from datetime import datetime, timezone
from test_cli_image import run_wwb
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
tmp_dir = tempfile.mkdtemp()


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _log(message: str) -> None:
    print(f"[{_ts()}] [wwb-reranking] {message}", flush=True)


def _require_optimum_cli() -> None:
    if shutil.which("optimum-cli") is None:
        pytest.skip("Missing required executable 'optimum-cli' for reranking export.")


OV_RERANK_MODELS = {
    ("cross-encoder/ms-marco-TinyBERT-L2-v2", "text-classification"),
    ("Qwen/Qwen3-Reranker-0.6B", "text-generation"),
}


def setup_module():
    _require_optimum_cli()
    _log(f"tmp_dir={tmp_dir}")
    for model_info in OV_RERANK_MODELS:
        model_id = model_info[0]
        task = model_info[1]
        MODEL_PATH = Path(tmp_dir, model_id.replace("/", "_"))
        _log(f"Export OpenVINO model: model_id={model_id} task={task} -> {MODEL_PATH}")
        start = time.perf_counter()
        result = subprocess.run(
            [
                "optimum-cli",
                "export",
                "openvino",
                "--model",
                model_id,
                MODEL_PATH,
                "--task",
                task,
                "--trust-remote-code",
            ],
            capture_output=True,
            text=True,
        )
        _log(f"optimum-cli export dt={time.perf_counter() - start:.3f}s rc={result.returncode}")
        assert result.returncode == 0


def teardown_module():
    logger.info("Remove models")
    shutil.rmtree(tmp_dir)


@pytest.mark.parametrize(("model_info"), OV_RERANK_MODELS)
def test_reranking_genai(model_info, tmp_path):
    if sys.platform == 'darwin':
        pytest.xfail("Ticket 175534")

    _log(f"Test params: model_info={model_info} tmp_path={tmp_path}")

    GT_FILE = Path(tmp_dir) / "gt.csv"
    model_id = model_info[0]
    MODEL_PATH = Path(tmp_dir) / model_id.replace("/", "_")

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
    _require_optimum_cli()
    _log(f"Test params: model_info={model_info} tmp_path={tmp_path}")
    GT_FILE = Path(tmp_dir) / "gt.csv"
    model_id = model_info[0]
    MODEL_PATH = Path(tmp_dir, model_id.replace("/", "_"))

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
