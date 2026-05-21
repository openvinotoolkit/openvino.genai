# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import sys

import pandas as pd
import pytest
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

from conftest import convert_text_model, run_wwb


def _convert_base(model_id, temp_path):
    from optimum.exporters.openvino.convert import export_tokenizer
    from huggingface_hub import snapshot_download

    model_local = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_local)
    base_model = OVModelForCausalLM.from_pretrained(model_local)
    base_model.save_pretrained(temp_path)
    tokenizer.save_pretrained(temp_path)
    export_tokenizer(tokenizer, temp_path)


agent_model_id = "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"
agent_model_path = convert_text_model(agent_model_id, "tiny_phi3_agent", _convert_base)


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
    if sys.platform == "darwin":
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


@pytest.mark.parametrize(
    ("dataset_name", "as_jsonl", "use_genai"),
    [
        ("messages_optimum.json", False, False),
        ("messages_optimum.jsonl", True, False),
        ("messages_genai.json", False, True),
        ("messages_genai.jsonl", True, True),
    ],
)
def test_text_agent_dataset_json_backend(dataset_name, as_jsonl, use_genai, tmp_path):
    if sys.platform == "darwin":
        pytest.xfail("Ticket 173169")

    dataset_path = tmp_path / dataset_name
    gt_path = tmp_path / f"gt_{dataset_name}.csv"
    _create_messages_dataset(dataset_path, as_jsonl=as_jsonl)

    output = run_wwb([
        "--base-model",
        agent_model_path,
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
        *(["--genai"] if use_genai else []),
    ])

    data = pd.read_csv(gt_path)
    assert len(data["prompts"].values) == 1
    assert "Say hello in one word." in data["prompts"].values[0]
    assert "Agent dataset summary" in output


@pytest.mark.parametrize(
    ("dataset_name", "as_jsonl", "use_genai"),
    [
        ("messages_target_data.json", False, False),
        ("messages_target_data.jsonl", True, False),
        ("messages_target_data_genai.json", False, True),
        ("messages_target_data_genai.jsonl", True, True),
    ],
)
def test_text_agent_target_data(dataset_name, as_jsonl, use_genai, tmp_path):
    if sys.platform == "darwin":
        pytest.xfail("Ticket 173169")

    dataset_path = tmp_path / dataset_name
    gt_path = tmp_path / f"gt_{dataset_name}.csv"
    output_dir = tmp_path / "target_data_output"
    target_data_path = output_dir / "target.csv"
    _create_messages_dataset(dataset_path, as_jsonl=as_jsonl)

    run_wwb([
        "--base-model",
        agent_model_path,
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
    ])

    output_with_models = run_wwb([
        "--target-model",
        agent_model_path,
        "--gt-data",
        gt_path,
        "--model-type",
        "text-agent",
        "--num-samples",
        "1",
        "--device",
        "CPU",
        "--output",
        output_dir,
    ])
    assert "Metrics for model" in output_with_models
    assert target_data_path.exists()

    output_with_target_data = run_wwb([
        "--gt-data",
        gt_path,
        "--target-data",
        target_data_path,
        "--model-type",
        "text-agent",
        "--num-samples",
        "1",
        "--device",
        "CPU",
        *(["--genai"] if use_genai else []),
    ])
    assert "Metrics for model" in output_with_target_data
