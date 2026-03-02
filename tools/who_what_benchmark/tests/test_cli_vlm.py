import pytest
import logging
import sys

from conftest import convert_model, run_wwb
from test_cli_image import get_similarity
from profile_utils import _log, _stage


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_test(model_id, model_type, optimum_threshold, genai_threshold, tmp_path):
    if sys.platform == 'darwin':
        pytest.xfail("Ticket 173169")
    if sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    _log(f"run_test: model_id={model_id} model_type={model_type} optimum_threshold={optimum_threshold} genai_threshold={genai_threshold}")
    GT_FILE = tmp_path / "gt.csv"
    with _stage("convert_model"):
        MODEL_PATH = convert_model(model_id)
    _log(f"Model converted to: {MODEL_PATH}")

    # Collect reference with HF model
    with _stage("run_wwb_hf_reference"):
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
    with _stage("run_wwb_optimum"):
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
    with _stage("run_wwb_genai"):
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
    with _stage("run_wwb_metrics_without_models"):
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
        ("optimum-intel-internal-testing/tiny-random-llava", "visual-text"),
    ],
)
def test_vlm_basic(model_id, model_type, tmp_path):
    _log(f"test_vlm_basic: model_id={model_id} model_type={model_type}")
    run_test(model_id, model_type, None, None, tmp_path)


@pytest.mark.nanollava
@pytest.mark.parametrize(
    ("model_id", "model_type", "optimum_threshold", "genai_threshold"),
    [
        ("qnguyen3/nanoLLaVA", "visual-text", 0.99, 0.88),
    ],
)
def test_vlm_nanollava(model_id, model_type, optimum_threshold, genai_threshold, tmp_path):
    _log(f"test_vlm_nanollava: model_id={model_id} model_type={model_type}")
    run_test(model_id, model_type, optimum_threshold, genai_threshold, tmp_path)


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [
        ("optimum-intel-internal-testing/tiny-random-qwen2vl", "visual-video-text"),
        ("optimum-intel-internal-testing/tiny-random-llava-next-video", "visual-video-text"),
    ],
)
def test_vlm_video(model_id, model_type, tmp_path):
    _log(f"test_vlm_video: model_id={model_id} model_type={model_type}")
    run_test(model_id, model_type, 0.8, 0.8, tmp_path)
