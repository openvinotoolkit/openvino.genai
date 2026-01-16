import subprocess  # nosec B404
import pytest
import logging
import sys
import os
from test_cli_image import run_wwb


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_test(model_id, model_type, optimum_threshold, genai_threshold, tmp_path, pruning_ratio, relevance_weight):
    if sys.platform == "darwin":
        pytest.xfail("Ticket 173169")
    if sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = tmp_path / model_id.replace("/", "_")

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
    assert result.returncode == 0

    # test cdpruner
    output = run_wwb(
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
            model_type,
            "--genai",
            "--output",
            tmp_path,
            "--pruning_ratio",
            pruning_ratio,
            "--relevance_weight",
            relevance_weight,
        ]
    )

    if pruning_ratio > 0 and pruning_ratio < 100:
        pruner_info = "[INFO]    Pruning Ratio: " + pruning_ratio + "%"
    elif pruning_ratio == 100:
        pruner_info = "Original visual tokens and pruned visual tokens are the same!"

    assert pruner_info in output


@pytest.mark.parametrize(
    ("model_id", "model_type", "pruning_ratio", "relevance_weight"),
    [
        ("optimum-intel-internal-testing/tiny-random-qwen2vl", "visual-text", "20", "0.8"),
        ("optimum-intel-internal-testing/tiny-random-qwen2vl", "visual-text", "100", "0.8"),
    ],
)
def test_pruner_basic(model_id, model_type, tmp_path, pruning_ratio, relevance_weight):
    env = os.environ.copy()
    env["OPENVINO_LOG_LEVEL"] = "7"
    run_test(model_id, model_type, None, None, tmp_path, pruning_ratio, relevance_weight)
