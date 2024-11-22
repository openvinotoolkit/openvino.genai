import subprocess  # nosec B404
import os
import shutil
import tempfile
import pandas as pd
import pytest
import logging

from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM, OVWeightQuantizationConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_wwb(args):
    logger.info(" ".join(["wwb"] + args))
    result = subprocess.run(["wwb"] + args, capture_output=True, text=True)
    logger.info(result)
    return result


model_id = "facebook/opt-125m"
tmp_dir = tempfile.mkdtemp()
base_model_path = os.path.join(tmp_dir, "opt125m")
target_model_path = os.path.join(tmp_dir, "opt125m_int8")


def setup_module():
    from optimum.exporters.openvino.convert import export_tokenizer

    logger.info("Create models")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = OVModelForCausalLM.from_pretrained(model_id)
    base_model.save_pretrained(base_model_path)
    tokenizer.save_pretrained(base_model_path)
    export_tokenizer(tokenizer, base_model_path)

    target_model = OVModelForCausalLM.from_pretrained(
        model_id, quantization_config=OVWeightQuantizationConfig(bits=8)
    )
    target_model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)
    export_tokenizer(tokenizer, target_model_path)


def teardown_module():
    logger.info("Remove models")
    shutil.rmtree(tmp_dir)


def test_text_target_model():
    result = run_wwb(
        [
            "--base-model",
            base_model_path,
            "--target-model",
            target_model_path,
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--model-type",
            "text",
        ]
    )

    assert result.returncode == 0
    assert "Metrics for model" in result.stderr
    assert "## Reference text" not in result.stderr


@pytest.fixture
def test_text_gt_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_name = os.path.join(temp_dir, "gt.csv")

        result = run_wwb(
            [
                "--base-model",
                base_model_path,
                "--gt-data",
                temp_file_name,
                "--dataset",
                "EleutherAI/lambada_openai,en",
                "--dataset-field",
                "text",
                "--split",
                "test",
                "--num-samples",
                "2",
                "--device",
                "CPU",
            ]
        )
        data = pd.read_csv(temp_file_name)

    assert result.returncode == 0
    assert len(data["questions"].values) == 2


def test_text_output_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        result = run_wwb(
            [
                "--base-model",
                base_model_path,
                "--gt-data",
                os.path.join(temp_dir, "gt.csv"),
                "--target-model",
                target_model_path,
                "--num-samples",
                "2",
                "--device",
                "CPU",
                "--output",
                temp_dir,
            ]
        )
        assert result.returncode == 0
        assert "Metrics for model" in result.stderr
        assert os.path.exists(os.path.join(temp_dir, "metrics_per_qustion.csv"))
        assert os.path.exists(os.path.join(temp_dir, "metrics.csv"))
        assert os.path.exists(os.path.join(temp_dir, "target.csv"))

        # test measurtement w/o models
        result = run_wwb(
            [
                "--gt-data",
                os.path.join(temp_dir, "gt.csv"),
                "--target-data",
                os.path.join(temp_dir, "target.csv"),
                "--num-samples",
                "2",
                "--device",
                "CPU",
            ]
        )
        assert result.returncode == 0
        assert "Metrics for model" in result.stderr


def test_text_verbose():
    result = run_wwb(
        [
            "--base-model",
            base_model_path,
            "--target-model",
            target_model_path,
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--verbose",
        ]
    )
    assert result.returncode == 0
    assert "## Diff " in result.stderr


def test_text_language_autodetect():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_name = os.path.join(temp_dir, "gt.csv")
        result = run_wwb(
            [
                "--base-model",
                "Qwen/Qwen2-0.5B",
                "--gt-data",
                temp_file_name,
                "--num-samples",
                "2",
                "--device",
                "CPU",
            ]
        )
        data = pd.read_csv(temp_file_name)

    assert result.returncode == 0
    assert "马克" in data["prompts"].values[0]


def test_text_hf_model():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_name = os.path.join(temp_dir, "gt.csv")
        result = run_wwb(
            [
                "--base-model",
                model_id,
                "--gt-data",
                temp_file_name,
                "--num-samples",
                "2",
                "--device",
                "CPU",
                "--hf",
            ]
        )
        data = pd.read_csv(temp_file_name)

    assert result.returncode == 0
    assert len(data["prompts"].values) == 2


def test_text_genai_model():
    result = run_wwb(
        [
            "--base-model",
            base_model_path,
            "--target-model",
            target_model_path,
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--genai",
        ]
    )
    assert result.returncode == 0
    assert "Metrics for model" in result.stderr
    assert "## Reference text" not in result.stderr
