import subprocess  # nosec B404
import os
import shutil
import sys
import pytest
import logging
import tempfile
from test_cli_image import run_wwb, get_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CACHE = tempfile.mkdtemp()
OV_VIDEO_MODELS = ["optimum-intel-internal-testing/tiny-random-ltx-video"]


def setup_module():
    for model_id in OV_VIDEO_MODELS:
        MODEL_PATH = os.path.join(MODEL_CACHE, model_id.replace("/", "_"))
        subprocess.run(
            ["optimum-cli", "export", "openvino", "--model", model_id, MODEL_PATH], capture_output=True, text=True
        )


def teardown_module():
    logger.info("Remove models")
    shutil.rmtree(MODEL_CACHE)


@pytest.mark.xfail(sys.platform == "darwin", reason="Not enough memory on macOS CI runners. Ticket CVS-179749")
@pytest.mark.xfail(sys.platform == "win32", reason="Access violation in OVLTXPipeline on Windows. Ticket CVS-179750")
@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [("optimum-intel-internal-testing/tiny-random-ltx-video", "text-to-video")],
)
def test_video_model_genai(model_id, model_type, tmp_path):
    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = os.path.join(MODEL_CACHE, model_id.replace("/", "_"))

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
            model_type,
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
        ]
    )
    assert GT_FILE.exists()
    assert (tmp_path / "reference").exists()

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
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
            "--output",
            tmp_path,
        ]
    )

    assert "Metrics for model" in output
    similarity = get_similarity(output)
    assert similarity >= 0.88
    assert (tmp_path / "target").exists()

    # test w/o models
    run_wwb(
        [
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
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
        ]
    )
