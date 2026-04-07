# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
import logging
import json
import sys

from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM, OVWeightQuantizationConfig

from test_cli_image import get_similarity
from conftest import convert_text_model, run_wwb


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model_id = "facebook/opt-125m"

# awq/gptq models are skipped for now: 180586
awq_model_id = "TitanML/tiny-mixtral-AWQ-4bit"
# model load failed - ticket: 178940, 180586
gptq_model_id = "ybelkada/opt-125m-gptq-4bit"


def _convert_base(model_id, temp_path):
    from optimum.exporters.openvino.convert import export_tokenizer
    from huggingface_hub import snapshot_download

    model_local = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_local)
    base_model = OVModelForCausalLM.from_pretrained(model_local)
    base_model.save_pretrained(temp_path)
    tokenizer.save_pretrained(temp_path)
    export_tokenizer(tokenizer, temp_path)


def _convert_int8(model_id, temp_path):
    from optimum.exporters.openvino.convert import export_tokenizer
    from huggingface_hub import snapshot_download

    model_local = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_local)
    target_model = OVModelForCausalLM.from_pretrained(
        model_local, quantization_config=OVWeightQuantizationConfig(bits=8)
    )
    target_model.save_pretrained(temp_path)
    tokenizer.save_pretrained(temp_path)
    export_tokenizer(tokenizer, temp_path)


base_model_path = convert_text_model(model_id, "opt125m", _convert_base)
target_model_path = convert_text_model(model_id, "opt125m_int8", _convert_int8)


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


@pytest.mark.parametrize(
    ("model_id"),
    [(model_id)],
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
    if sys.platform == "darwin":
        pytest.xfail(
            "Continuous batching backend requires PagedAttention operation support, which is available on x86_64 or ARM64 platforms only"
        )
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
    if sys.platform == "darwin":
        pytest.xfail(
            "Continuous batching backend requires PagedAttention operation support, which is available on x86_64 or ARM64 platforms only"
        )

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


@pytest.mark.parametrize(
    ("model_id"),
    [("optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM")],
)
def test_text_chat_model(model_id, tmp_path):
    if sys.platform == "darwin":
        pytest.xfail("Ticket 183495")

    SIMILARITY_THRESHOLD = 0.9
    temp_file_name = tmp_path / "gt.csv"
    chat_model_path = convert_text_model(model_id, model_id.split("/")[1], _convert_base)

    run_wwb(
        [
            "--base-model",
            model_id,
            "--gt-data",
            temp_file_name,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--model-type",
            "text-chat",
            "--hf",
            "--max_new_tokens",
            "10",
        ]
    )

    outputs_path = tmp_path / "optimum"
    output = run_wwb(
        [
            "--target-model",
            chat_model_path,
            "--gt-data",
            temp_file_name,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--model-type",
            "text-chat",
            "--output",
            outputs_path,
            "--max_new_tokens",
            "10",
        ]
    )
    assert "Metrics for model" in output
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert (outputs_path / "target.csv").exists()

    similarity = get_similarity(output)
    assert similarity >= SIMILARITY_THRESHOLD

    outputs_path = tmp_path / "genai"
    output = run_wwb(
        [
            "--target-model",
            chat_model_path,
            "--gt-data",
            temp_file_name,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--model-type",
            "text-chat",
            "--genai",
            "--output",
            outputs_path,
            "--max_new_tokens",
            "10",
        ]
    )
    assert "Metrics for model" in output
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert (outputs_path / "target.csv").exists()

    similarity = get_similarity(output)
    assert similarity >= SIMILARITY_THRESHOLD

def _create_messages_dataset(path, as_jsonl=False):
    records = [
        {
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "tools": [],
        }
    ]
    if as_jsonl:
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(records), encoding="utf-8")


@pytest.mark.parametrize(
    ("dataset_name", "as_jsonl"),
    [
        ("messages.json", False),
        ("messages.jsonl", True),
    ],
)
def test_text_agent_dataset_json(dataset_name, as_jsonl, tmp_path):
    if sys.platform == 'darwin':
        pytest.xfail("Ticket 173169")

    dataset_path = tmp_path / dataset_name
    gt_path = tmp_path / f"gt_{dataset_name}.csv"
    _create_messages_dataset(dataset_path, as_jsonl=as_jsonl)

    output = run_wwb([
        "--base-model",
        "Qwen/Qwen2-0.5B",
        "--gt-data",
        gt_path,
        "--dataset",
        dataset_path,
        "--model-type",
        "text-agent",
        "--num-samples",
        "1",
        "--device",
        "CPU",
        "--hf",
    ])

    data = pd.read_csv(gt_path)
    assert len(data["prompts"].values) == 1
    assert "Say hello in one word." in data["prompts"].values[0]
    assert "Agent dataset summary" in output