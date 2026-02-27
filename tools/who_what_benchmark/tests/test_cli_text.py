import os
import shutil
import tempfile
import pandas as pd
import pytest
import logging
import json
import sys

from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM, OVWeightQuantizationConfig

from conftest import run_wwb


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model_id = "facebook/opt-125m"
tmp_dir = tempfile.mkdtemp()
base_model_path = os.path.join(tmp_dir, "opt125m")
target_model_path = os.path.join(tmp_dir, "opt125m_int8")

gptq_model_id = "ybelkada/opt-125m-gptq-4bit"
awq_model_id = "TitanML/tiny-mixtral-AWQ-4bit"


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


@pytest.mark.skipif((sys.platform == "darwin"), reason='173169')
def test_text_target_model():
    run_wwb([
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
    ])


@pytest.fixture
def test_text_gt_data(tmp_path):
    temp_file_name = tmp_path / "gt.csv"
    run_wwb([
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
    ])
    data = pd.read_csv(temp_file_name)
    assert len(data["questions"].values) == 2


def test_text_output_directory(tmp_path):
    if sys.platform == 'darwin':
        pytest.xfail("Ticket 173169")
    temp_file_name = tmp_path / "gt.csv"
    output = run_wwb([
        "--base-model",
        base_model_path,
        "--gt-data",
        temp_file_name,
        "--target-model",
        target_model_path,
        "--num-samples",
        "2",
        "--device",
        "CPU",
        "--output",
        tmp_path,
    ])
    assert "Metrics for model" in output
    assert (tmp_path / "metrics_per_question.csv").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "target.csv").exists()

    measurement_without_models = run_wwb([
        "--gt-data",
        temp_file_name,
        "--target-data",
        tmp_path / "target.csv",
        "--num-samples",
        "2",
        "--device",
        "CPU",
    ])
    assert "Metrics for model" in measurement_without_models


def test_text_verbose():
    if sys.platform == 'darwin':
        pytest.xfail("Ticket 173169")
    output = run_wwb([
        "--base-model",
        base_model_path,
        "--target-model",
        target_model_path,
        "--num-samples",
        "2",
        "--device",
        "CPU",
        "--verbose",
    ])
    assert "## Diff:" in output


def test_text_language(tmp_path):
    temp_file_name = tmp_path / "gt.csv"
    run_wwb([
        "--base-model",
        "Qwen/Qwen2-0.5B",
        "--gt-data",
        temp_file_name,
        "--num-samples",
        "2",
        "--device",
        "CPU",
        "--language",
        "cn",
    ])
    data = pd.read_csv(temp_file_name)
    assert "马克" in data["prompts"].values[0]


hf_model_scope = [
    (model_id),
]
if sys.platform != 'darwin' and sys.platform != 'win32':
    hf_model_scope += [
        # model load failed in optimum, ticket: 178940
        # (gptq_model_id),
        (awq_model_id),
    ]


@pytest.mark.parametrize(
    ("model_id"),
    hf_model_scope,
)
def test_text_hf_model(model_id, tmp_path):
    temp_file_name = tmp_path / "gt.csv"
    run_wwb([
        "--base-model",
        model_id,
        "--gt-data",
        temp_file_name,
        "--num-samples",
        "1",
        "--device",
        "CPU",
        "--hf",
    ])
    data = pd.read_csv(temp_file_name)
    assert len(data["prompts"].values) == 1


def test_text_genai_model():
    if sys.platform == 'darwin':
        pytest.xfail("Ticket 173169")
    output = run_wwb([
        "--base-model",
        base_model_path,
        "--target-model",
        target_model_path,
        "--num-samples",
        "2",
        "--device",
        "CPU",
        "--genai",
    ])
    assert "Metrics for model" in output
    assert "## Reference text" not in output


def test_text_genai_cb_model(tmp_path):
    if sys.platform == 'darwin':
        pytest.xfail("Ticket 173169")
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        config = {
            "dynamic_split_fuse": True,
            "use_cache_eviction": True,
            "cache_eviction_config":
            {
                "start_size": 32,
                "recent_size": 32,
                "max_cache_size": 96
            }
        }
        json.dump(config, f)

    ov_config_path = tmp_path / "ov_config.json"
    with open(ov_config_path, "w") as f:
        config = {
            "KV_CACHE_PRECISION": "f16",
            "ATTENTION_BACKEND": "PA"
        }
        json.dump(config, f)

    output = run_wwb([
        "--base-model",
        base_model_path,
        "--target-model",
        target_model_path,
        "--num-samples",
        "2",
        "--device",
        "CPU",
        "--genai",
        "--cb-config",
        config_path,
        "--ov-config",
        ov_config_path
    ])
    assert "Metrics for model" in output
    assert "## Reference text" not in output
    assert "INFO:whowhatbench.model_loaders:OpenVINO Config: {'KV_CACHE_PRECISION': 'f16', 'ATTENTION_BACKEND': 'PA'}" in output


def test_text_genai_json_string_config():
    if sys.platform == 'darwin':
        pytest.xfail("Ticket 173169")

    cb_json_string = "{\"max_num_batched_tokens\": 4096}"
    ov_json_string = "{\"KV_CACHE_PRECISION\":\"f16\", \"ATTENTION_BACKEND\": \"PA\"}"

    output = run_wwb([
        "--base-model",
        base_model_path,
        "--target-model",
        target_model_path,
        "--num-samples",
        "2",
        "--device",
        "CPU",
        "--genai",
        "--cb-config",
        cb_json_string,
        "--ov-config",
        ov_json_string
    ])

    # Test with WWB log info to make sure the configurations are passed from strings to the GenAI APIs
    assert "INFO:whowhatbench.wwb:cb_config: {'max_num_batched_tokens': 4096}" in output
    assert "INFO:whowhatbench.model_loaders:OpenVINO Config: {'KV_CACHE_PRECISION': 'f16', 'ATTENTION_BACKEND': 'PA'}" in output
