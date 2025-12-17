import subprocess  # nosec B404
import sys
import pytest
import logging
import shutil
import time
from datetime import datetime, timezone
from test_cli_image import run_wwb


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _log(message: str) -> None:
    print(f"[{_ts()}] [wwb-embeddings] {message}", flush=True)


def _require_optimum_cli() -> None:
    if shutil.which("optimum-cli") is None:
        pytest.skip("Missing required executable 'optimum-cli' for embeddings export.")


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
    _require_optimum_cli()
    _log(f"Test params: model_id={model_id} model_type={model_type} tmp_path={tmp_path}")
    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = tmp_path / model_id.replace("/", "_")

    start = time.perf_counter()
    result = subprocess.run(
        [
            "optimum-cli",
            "export",
            "openvino",
            "-m",
            model_id,
            MODEL_PATH,
            "--task",
            "feature-extraction",
            "--trust-remote-code",
        ],
        capture_output=True,
        text=True,
    )
    _log(f"optimum-cli export dt={time.perf_counter() - start:.3f}s rc={result.returncode} MODEL_PATH={MODEL_PATH}")
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
    _require_optimum_cli()
    _log(f"Test params: model_id={model_id} model_type={model_type} batch_size={batch_size} tmp_path={tmp_path}")
    GT_FILE = tmp_path / f"gt_batch_{batch_size}.csv"
    MODEL_PATH = tmp_path / model_id.replace("/", "_")

    start = time.perf_counter()
    result = subprocess.run(
        [
            "optimum-cli",
            "export",
            "openvino",
            "-m",
            model_id,
            MODEL_PATH,
            "--task",
            "feature-extraction",
            "--trust-remote-code",
        ],
        capture_output=True,
        text=True,
    )
    _log(f"optimum-cli export dt={time.perf_counter() - start:.3f}s rc={result.returncode} MODEL_PATH={MODEL_PATH}")
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
