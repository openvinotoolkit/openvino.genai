import subprocess  # nosec B404
import os
import shutil
import pytest
import logging
import tempfile


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_wwb(args):
    logger.info(" ".join(["TRANSFOREMRS_VERBOSITY=debug wwb"] + args))
    result = subprocess.run(["wwb"] + args, capture_output=True, text=True)
    logger.info(result)
    return result


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [
        ("katuni4ka/tiny-random-llava", "visual-text"),
    ],
)
def test_vlm_basic(model_id, model_type):
    with tempfile.TemporaryDirectory() as temp_dir:
        GT_FILE = os.path.join(temp_dir, "gt.csv")
        MODEL_PATH = os.path.join(temp_dir, model_id.replace("/", "--"))

        result = subprocess.run(["optimum-cli", "export",
                                 "openvino", "-m", model_id,
                                 MODEL_PATH, "--task",
                                 "image-text-to-text",
                                 "--trust-remote-code"],
                                capture_output=True,
                                text=True,
                                )
        assert result.returncode == 0

        # Collect reference with HF model
        wwb_args = [
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
        ]
        result = run_wwb(wwb_args)
        assert result.returncode == 0

        # test Optimum
        wwb_args = [
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
        ]
        result = run_wwb(wwb_args)
        assert result.returncode == 0

        # test GenAI
        wwb_args = [
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
            "target",
        ]
        result = run_wwb(wwb_args)
        assert result.returncode == 0

        # test w/o models
        wwb_args = [
            "--target-data",
            "target/target.csv",
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
        ]
        result = run_wwb(wwb_args)
        assert result.returncode == 0
        shutil.rmtree("reference", ignore_errors=True)
        shutil.rmtree("target", ignore_errors=True)
        shutil.rmtree(MODEL_PATH, ignore_errors=True)
