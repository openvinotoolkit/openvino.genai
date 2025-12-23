import subprocess  # nosec B404
import pytest
import logging
import sys
import shutil
import time
from datetime import datetime, timezone
from test_cli_image import run_wwb, get_similarity


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _log(message: str) -> None:
    print(f"[{_ts()}] [wwb-vlm] {message}", flush=True)


def _require_optimum_cli() -> None:
    if shutil.which("optimum-cli") is None:
        pytest.skip("Missing required executable 'optimum-cli' for VLM export.")


def run_test(model_id, model_type, optimum_threshold, genai_threshold, tmp_path):
    if sys.platform == 'darwin':
        pytest.xfail("Ticket 173169")
    _require_optimum_cli()
    _log(
        "Test params: "
        f"model_id={model_id} model_type={model_type} optimum_threshold={optimum_threshold} genai_threshold={genai_threshold} tmp_path={tmp_path}"
    )
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
            "image-text-to-text",
            "--trust-remote-code",
        ],
        capture_output=True,
        text=True,
    )
    _log(f"optimum-cli export dt={time.perf_counter() - start:.3f}s rc={result.returncode} MODEL_PATH={MODEL_PATH}")
    assert result.returncode == 0

    # Collect reference with HF model
    _log("Collecting reference (HF) via wwb")
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
    _log("Testing Optimum-exported OpenVINO model via wwb")
    output = run_wwb([
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
    if optimum_threshold is not None:
        similarity = get_similarity(output)
        assert similarity >= optimum_threshold

    # test GenAI
    _log("Testing GenAI backend via wwb")
    output = run_wwb([
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
    if genai_threshold is not None:
        similarity = get_similarity(output)
        assert similarity >= genai_threshold

    # test w/o models
    _log("Testing wwb without models (target-data only)")
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
    ("model_id", "model_type"),
    [
        ("katuni4ka/tiny-random-llava", "visual-text"),
    ],
)
def test_vlm_basic(model_id, model_type, tmp_path):
    run_test(model_id, model_type, None, None, tmp_path)


@pytest.mark.nanollava
@pytest.mark.parametrize(
    ("model_id", "model_type", "optimum_threshold", "genai_threshold"),
    [
        ("qnguyen3/nanoLLaVA", "visual-text", 0.99, 0.88),
    ],
)
def test_vlm_nanollava(model_id, model_type, optimum_threshold, genai_threshold, tmp_path):
    run_test(model_id, model_type, optimum_threshold, genai_threshold, tmp_path)


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [
        ("katuni4ka/tiny-random-qwen2vl", "visual-video-text"),
        ("katuni4ka/tiny-random-llava-next-video", "visual-video-text"),
    ],
)
def test_vlm_video(model_id, model_type, tmp_path):
    run_test(model_id, model_type, 0.8, 0.8, tmp_path)
