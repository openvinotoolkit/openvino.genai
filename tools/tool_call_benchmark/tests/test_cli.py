# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI smoke test on a real model (skipped when TCB_TEST_MODEL is unset)."""

import json
import os
import subprocess  # nosec B404
import sys

import pytest

TEST_MODEL = os.environ.get("TCB_TEST_MODEL", "")


@pytest.mark.skipif(not TEST_MODEL or not os.path.isdir(TEST_MODEL),
                    reason="set TCB_TEST_MODEL to an OpenVINO IR dir to run")
def test_cli_three_cases(tmp_path):
    tool_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = tool_dir + os.pathsep + env.get("PYTHONPATH", "")
    out_dir = str(tmp_path / "out")
    result = subprocess.run(  # nosec B603
        [sys.executable, "-m", "toolcallbench.tcb",
         "--model", TEST_MODEL, "--case-ids", "A01,A02,A03",
         "--max-new-tokens", "256", "--output", out_dir],
        capture_output=True, text=True, timeout=1200, env=env)
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.load(open(os.path.join(out_dir, "report.json")))
    assert report["verdict"] == "INCOMPLETE"  # subset of the dataset
    assert len(report["results"]) == 3
    assert report["run"]["dataset_version"] == "coding_agent_v1"
    cases = [json.loads(l) for l in open(os.path.join(out_dir, "cases.jsonl"))]
    assert [c["id"] for c in cases] == ["A01", "A02", "A03"]
