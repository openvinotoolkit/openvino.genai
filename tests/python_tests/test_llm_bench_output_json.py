# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "iterations,expected_latency",
    [
        ([(0, 0, 100), (0, 1, 200), (1, 0, 10), (1, 1, 20)], 15),
        ([(0, 0, 100), (1, 0, 10), (0, 1, 200), (1, 1, 20)], 15),
        ([(0, 0, 100), (1, 0, 10), (2, 0, 20)], 15),
        ([(0, 0, 100), (0, 1, 200)], None),
    ],
    ids=["iteration-order", "prompt-order", "single-prompt", "warmup-only"],
)
def test_json_averages_exclude_all_warmup_iterations(monkeypatch, tmp_path, iterations, expected_latency):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools" / "llm_bench"))
    from llm_bench_utils.gen_output_data import gen_iterate_data
    from llm_bench_utils.output_json import write_result

    iteration_data = [
        gen_iterate_data(
            iter_idx=iteration,
            prompt_idx=prompt,
            in_size=1,
            infer_count=1,
            out_size=1,
            gen_time=latency / 1000,
            latency=latency,
            res_md5=[],
        )
        for iteration, prompt, latency in iterations
    ]
    report_path = tmp_path / "report.json"
    write_result(
        report_path,
        "test-model",
        "ov",
        "CPU",
        {"num_beams": 1, "batch_size": 1},
        iteration_data,
        0,
        "FP32",
        {},
        None,
    )

    perfdata = json.loads(report_path.read_text())["perfdata"]
    assert [entry["iteration"] for entry in perfdata["results"]] == [entry[0] for entry in iterations]
    if expected_latency is None:
        assert "results_averaged" not in perfdata
    else:
        assert perfdata["results_averaged"]["latency"] == pytest.approx(expected_latency)
        assert perfdata["results_averaged"]["generation_time"] == pytest.approx(expected_latency / 1000)
