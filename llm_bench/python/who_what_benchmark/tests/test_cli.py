import subprocess
import os
import shutil
import tempfile
import pandas as pd
import pytest

from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM, OVWeightQuantizationConfig


def run_wwb(args):
    print(" ".join(["wwb"] + args))
    result = subprocess.run(
        ["wwb"] + args,
        capture_output=True,
        text=True
    )
    print(result)
    return result

model_id = "facebook/opt-125m"
tmp_dir = tempfile.mkdtemp()
base_model_path = os.path.join(tmp_dir, "opt125m")
target_model_path = os.path.join(tmp_dir, "opt125m_int8")

def setup_module():
    print("Create models")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = OVModelForCausalLM.from_pretrained(model_id)
    base_model.save_pretrained(base_model_path)
    tokenizer.save_pretrained(base_model_path)

    target_model = OVModelForCausalLM.from_pretrained(model_id, 
        quantization_config=OVWeightQuantizationConfig(bits=8))
    target_model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)

def teardown_module():
    print("Remove models")
    shutil.rmtree(tmp_dir)

def test_target_model():
    result = run_wwb([
        "--base-model", base_model_path,
        "--target-model", target_model_path,
        "--num-samples", "2",
        "--device", "CPU"
    ])
    assert result.returncode == 0
    assert "Metrics for model" in result.stdout
    assert not "## Reference text" in result.stdout

@pytest.fixture
def test_gt_data():
    # with tempfile.TemporaryDirectory() as tmp:
    #     temp_file = os.path.join(tmp.mkdtemp(), "gt.csv")
    with tempfile.NamedTemporaryFile(suffix=".csv")  as tmpfile:
        temp_file_name = tmpfile.name

    result = run_wwb([
        "--base-model", base_model_path,
        "--gt-data", temp_file_name,
        "--dataset", "EleutherAI/lambada_openai,en",
        "--dataset-field", "text",
        "--split", "test",
        "--num-samples", "2",
        "--device", "CPU"
    ])
    print(result)
    if os.path.exists(temp_file_name):
        print("GT file was created")
    else:
        print("No GT file was created")
    import time
    time.sleep(1)
    data = pd.read_csv(temp_file_name)
    os.remove(temp_file_name)
        
    assert result.returncode == 0
    assert len(data["questions"].values) == 2

def test_output_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        result = run_wwb([
            "--base-model", base_model_path,
            "--target-model", target_model_path,
            "--num-samples", "2",
            "--device", "CPU",
            "--output", temp_dir
        ])
        assert result.returncode == 0
        assert "Metrics for model" in result.stdout
        assert os.path.exists(os.path.join(temp_dir, "metrics_per_qustion.csv"))
        assert os.path.exists(os.path.join(temp_dir, "metrics.csv"))

def test_verbose():
    result = run_wwb([
        "--base-model", base_model_path,
        "--target-model", target_model_path,
        "--num-samples", "2",
        "--device", "CPU",
        "--verbose"
    ])
    assert result.returncode == 0
    assert "## Reference text" in result.stdout

def test_language_autodetect():
    with tempfile.NamedTemporaryFile(suffix=".csv")  as tmpfile:
        temp_file_name = tmpfile.name

    result = run_wwb([
        "--base-model", "Qwen/Qwen2-0.5B",
        "--gt-data", temp_file_name,
        "--num-samples", "2",
        "--device", "CPU"
    ])
    data = pd.read_csv(temp_file_name)
    os.remove(temp_file_name)
        
    assert result.returncode == 0
    assert "马克" in data["questions"].values[0]