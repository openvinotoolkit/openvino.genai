import subprocess  # nosec B404
import pytest
import logging
import sys
import os
from test_cli_image import run_wwb
from whowhatbench.utils import get_json_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_test(model_id, model_type, optimum_threshold, genai_threshold, tmp_path, generation_config_string):
    if sys.platform == 'darwin':
        pytest.xfail("Ticket 173169")
    if sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = tmp_path / model_id.replace("/", "_")

    result = subprocess.run(["optimum-cli", "export",
                             "openvino", "-m", model_id,
                             MODEL_PATH, "--task",
                             "image-text-to-text",
                             "--trust-remote-code"],
                            capture_output=True,
                            text=True,
                            )
    assert result.returncode == 0

    # test cdpruner
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
        "--generation-config",
        generation_config_string
    ])

    gen_config = get_json_config(generation_config_string)
    pruning_ratio = gen_config.get("pruning_ratio")
    if pruning_ratio > 0 and pruning_ratio < 100:
        pruner_info = "[INFO]    Pruning Ratio: " +  pruning_ratio
    elif pruning_ratio == 100:
        pruner_info = "Original visual tokens and pruned visual tokens are the same!"

    assert pruner_info in output


@pytest.mark.parametrize(
    ("model_id", "model_type", "generation_config_string"),
    [
        ("optimum-intel-internal-testing/tiny-random-qwen2vl", "visual-text", "{\"pruning_ratio\": 10}"),
        ("optimum-intel-internal-testing/tiny-random-qwen2vl", "visual-text", "{\"pruning_ratio\": 90}"),
        ("optimum-intel-internal-testing/tiny-random-qwen2vl", "visual-text", "{\"pruning_ratio\": 100}"),
    ],
)
def test_pruner_basic(model_id, model_type, tmp_path, generation_config_string):
    env = os.environ.copy()
    env["OPENVINO_LOG_LEVEL"] = "7"
    run_test(model_id, model_type, None, None, tmp_path, generation_config_string)
