# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import convert_model, run_wwb
from test_cli_image import get_similarity


agent_model_id = "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"
agent_model_path = convert_model(agent_model_id)
real_messages_path = Path(__file__).resolve().parents[1] / "whowhatbench" / "prompts" / "messages_500.jsonl"


def _create_messages_dataset(path, as_jsonl=False, use_real_prompt=False):
    if use_real_prompt:
        record = json.loads(real_messages_path.read_text(encoding="utf-8").splitlines()[0])
    else:
        record = {
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "tools": [],
        }

    if as_jsonl:
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    else:
        path.write_text(json.dumps([record]), encoding="utf-8")

    return record["messages"][0]["content"]


@pytest.mark.parametrize(
    ("dataset_name", "as_jsonl", "use_real_prompt"),
    [
        ("messages.json", False, False),
        ("messages.jsonl", True, False),
        ("messages_real.jsonl", True, True),
    ],
)
def test_text_agent_end_to_end(dataset_name, as_jsonl, use_real_prompt, tmp_path):
    if sys.platform == "darwin":
        pytest.xfail("Ticket 173169")

    similarity_threshold = 0.9

    dataset_path = tmp_path / dataset_name
    gt_path = tmp_path / f"gt_{dataset_name}.csv"
    optimum_output_dir = tmp_path / "optimum_output"
    genai_output_dir = tmp_path / "genai_output"
    optimum_target_data_path = optimum_output_dir / "target.csv"
    expected_prompt = _create_messages_dataset(dataset_path, as_jsonl=as_jsonl, use_real_prompt=use_real_prompt)

    hf_output = run_wwb([
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
    assert expected_prompt in data["prompts"].values[0]
    assert "Agent dataset summary" in hf_output

    optimum_output = run_wwb([
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
        optimum_output_dir,
    ])
    assert "Metrics for model" in optimum_output
    assert get_similarity(optimum_output) >= similarity_threshold
    assert (optimum_output_dir / "metrics_per_question.csv").exists()
    assert (optimum_output_dir / "metrics.csv").exists()
    assert optimum_target_data_path.exists()

    genai_output = run_wwb([
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
        genai_output_dir,
        "--genai",
    ])
    assert "Metrics for model" in genai_output
    assert get_similarity(genai_output) >= similarity_threshold
    assert (genai_output_dir / "metrics_per_question.csv").exists()
    assert (genai_output_dir / "metrics.csv").exists()
    assert (genai_output_dir / "target.csv").exists()

    target_data_output = run_wwb([
        "--gt-data",
        gt_path,
        "--target-data",
        optimum_target_data_path,
        "--model-type",
        "text-agent",
        "--num-samples",
        "1",
        "--device",
        "CPU",
    ])
    assert "Metrics for model" in target_data_output
    assert get_similarity(target_data_output) >= similarity_threshold
