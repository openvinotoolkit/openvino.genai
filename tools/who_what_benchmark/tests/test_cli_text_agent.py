# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from argparse import Namespace
import sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import convert_model, run_wwb
from test_cli_image import get_similarity


real_messages_path = Path(__file__).resolve().parents[1] / "whowhatbench" / "prompts" / "agent_short.jsonl"


def test_load_prompts_rejects_local_json_for_non_agent(tmp_path):
    from whowhatbench.wwb import load_prompts

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"text": "hello"}\n', encoding="utf-8")

    args = Namespace(dataset=str(dataset_path), split=None, dataset_field="text")

    with pytest.raises(
        ValueError, match="supported only for --model-type text-agent or --model-type visual-text-agent"
    ):
        load_prompts(args)


@pytest.mark.skipif(sys.platform == "win32", reason="lin/mac specific path is checking")
def test_resolve_json_dataset_path_supports_home_shorthand(tmp_path, monkeypatch):
    from whowhatbench.utils import resolve_json_dataset_path

    dataset_file = tmp_path / "data.jsonl"
    dataset_file.write_text('{"messages": [], "tools": []}\n', encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = resolve_json_dataset_path("~/data.jsonl")

    assert resolved == str(dataset_file)


@pytest.mark.parametrize(
    "relative_path",
    [
        "nested/data.jsonl",
        "./nested/data.jsonl",
        r"nested\\data.jsonl",
        r".\\nested\\data.jsonl",
    ],
)
def test_resolve_json_dataset_path_supports_relative_paths(tmp_path, monkeypatch, relative_path):
    from whowhatbench.utils import resolve_json_dataset_path

    if "\\" in relative_path and sys.platform != "win32":
        pytest.skip("Windows-style relative path case")

    dataset_file = tmp_path / "nested" / "data.jsonl"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_file.write_text('{"messages": [], "tools": []}\n', encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    resolved = resolve_json_dataset_path(relative_path)

    assert resolved == str(dataset_file.resolve())


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
    ("model_id", "model_type", "dataset_name", "as_jsonl", "use_real_prompt", "similarity_threshold"),
    [
        (
            "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM",
            "text-agent",
            "messages.json",
            False,
            False,
            0.9,
        ),
        (
            "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM",
            "text-agent",
            "messages.jsonl",
            True,
            False,
            0.9,
        ),
        (
            "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM",
            "text-agent",
            "messages_real.jsonl",
            True,
            True,
            0.9,
        ),
        (
            "optimum-intel-internal-testing/tiny-random-llava",
            "visual-text-agent",
            "messages_vlm.jsonl",
            True,
            False,
            None,
        ),
    ],
)
def test_text_agent_end_to_end(
    model_id,
    model_type,
    dataset_name,
    as_jsonl,
    use_real_prompt,
    similarity_threshold,
    tmp_path,
):
    if sys.platform == "darwin":
        pytest.xfail("Ticket 173169")
    if model_type == "visual-text-agent" and sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    model_path = convert_model(model_id)

    dataset_path = tmp_path / dataset_name
    gt_path = tmp_path / f"gt_{dataset_name}.csv"
    optimum_output_dir = tmp_path / "optimum_output"
    genai_output_dir = tmp_path / "genai_output"
    optimum_target_data_path = optimum_output_dir / "target.csv"
    expected_prompt = _create_messages_dataset(dataset_path, as_jsonl=as_jsonl, use_real_prompt=use_real_prompt)
    extra_args = []
    if model_type == "visual-text-agent":
        chat_template_path = tmp_path / "visual_text_agent_chat_template.jinja"
        chat_template_path.write_text(
            """{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant:""",
            encoding="utf-8",
        )
        extra_args = ["--chat-template-source", str(chat_template_path)]

    hf_output = run_wwb(
        [
            "--base-model",
            model_id,
            "--gt-data",
            gt_path,
            "--dataset",
            dataset_path,
            "--model-type",
            model_type,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--hf",
        ]
        + extra_args
    )

    data = pd.read_csv(gt_path)
    assert len(data["prompts"].values) == 1
    prompt_value = data["prompts"].values[0]
    prompt_path = Path(prompt_value)
    if prompt_path.exists():
        prompt_value = prompt_path.read_text(encoding="utf-8")
    assert expected_prompt in prompt_value
    assert "Text-agent dataset selected from --dataset:" in hf_output

    optimum_output = run_wwb(
        [
            "--target-model",
            model_path,
            "--gt-data",
            gt_path,
            "--dataset",
            dataset_path,
            "--model-type",
            model_type,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--output",
            optimum_output_dir,
        ]
        + extra_args
    )
    assert "Metrics for model" in optimum_output
    if similarity_threshold is not None:
        assert get_similarity(optimum_output) >= similarity_threshold
    assert (optimum_output_dir / "metrics_per_question.csv").exists()
    assert (optimum_output_dir / "metrics.csv").exists()
    assert optimum_target_data_path.exists()

    genai_output = run_wwb(
        [
            "--target-model",
            model_path,
            "--gt-data",
            gt_path,
            "--dataset",
            dataset_path,
            "--model-type",
            model_type,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--output",
            genai_output_dir,
            "--genai",
        ]
        + extra_args
    )
    assert "Metrics for model" in genai_output
    if similarity_threshold is not None:
        assert get_similarity(genai_output) >= similarity_threshold
    assert (genai_output_dir / "metrics_per_question.csv").exists()
    assert (genai_output_dir / "metrics.csv").exists()
    assert (genai_output_dir / "target.csv").exists()

    target_data_output = run_wwb(
        [
            "--gt-data",
            gt_path,
            "--target-data",
            optimum_target_data_path,
            "--model-type",
            model_type,
            "--num-samples",
            "1",
            "--device",
            "CPU",
        ]
    )
    assert "Metrics for model" in target_data_output
    if similarity_threshold is not None:
        assert get_similarity(target_data_output) >= similarity_threshold
