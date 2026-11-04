# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "task,unit",
    [
        ("text_gen_chat", "ms/token"),
        ("text_gen", "ms"),
        ("visual_text_gen_chat", "ms"),
    ],
)
@pytest.mark.parametrize("first_latency", [-1, 0, 2.5], ids=["unavailable", "zero", "measured"])
def test_csv_first_latency_columns_match_the_header(monkeypatch, tmp_path, task, unit, first_latency):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools" / "llm_bench"))
    from llm_bench_utils.gen_output_data import gen_iterate_data
    from llm_bench_utils.memory_monitor import MemoryMonitorHandler
    from llm_bench_utils.output_csv import write_result

    data = []
    for iteration in [0, 1]:
        entry = gen_iterate_data(
            iter_idx=iteration,
            prompt_idx=0,
            chat_idx=0 if "chat" in task else "",
            in_size=1,
            infer_count=1,
            out_size=1,
            gen_time=0.01,
            latency=10,
            res_md5=[],
        )
        entry["first_token_latency"] = first_latency
        entry["first_token_infer_latency"] = first_latency
        data.append(entry)
    report = tmp_path / "report.csv"
    write_result(
        report,
        "test-model",
        "ov",
        "CPU",
        {"num_beams": 1, "batch_size": 1, "use_case": SimpleNamespace(task=task)},
        data,
        0,
        "FP32",
        {},
        MemoryMonitorHandler(SimpleNamespace(memory_consumption=0, memory_consumption_cooldown=0)),
    )

    with report.open(newline="") as report_file:
        rows = list(csv.DictReader(report_file))
    measurements = [row for row in rows if row["iteration"] in ["0", "1", "avg", "mini", "median"]]
    assert len(measurements) == 5
    for row in measurements:
        for name in [f"1st_latency({unit})", f"1st_infer_latency({unit})"]:
            if first_latency < 0:
                assert row[name] == "NA"
            else:
                assert float(row[name]) == first_latency
