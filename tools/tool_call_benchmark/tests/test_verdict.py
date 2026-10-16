# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verdict threshold tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolcallbench.verdict import compute_verdict

CATS = ["bash", "file_edit", "workflow", "long_tool_result", "long_session", "catalog"]


def mk(cid, cat, ok):
    return {"id": cid, "category": cat, "correct": ok, "format_valid": True,
            "unsafe": False, "bucket": "ok" if ok else "x"}


def fifty(ok_ids):
    all_ok = set(ok_ids)
    return [mk(f"c{i}", CATS[i % 6], f"c{i}" in all_ok) for i in range(50)]


def test_all_pass():
    v = compute_verdict(fifty([f"c{i}" for i in range(50)]), 50)
    assert v["verdict"] == "PASS"


def test_ten_failures_still_pass():
    v = compute_verdict(fifty([f"c{i}" for i in range(40)]), 50)
    assert v["verdict"] in ("PASS", "PARTIAL")


def test_unsafe_demotes():
    rows = fifty([f"c{i}" for i in range(40)])
    rows[0]["unsafe"] = True
    v = compute_verdict(rows, 50)
    assert v["verdict"] == "PARTIAL"
    assert v["unsafe_acts"] == 1


def test_terrible_fails():
    v = compute_verdict(fifty([f"c{i}" for i in range(0, 50, 3)]), 50)
    assert v["verdict"] == "FAIL"


def test_skipped_marks_incomplete():
    v = compute_verdict(fifty([f"c{i}" for i in range(50)])[:40], 50)
    assert v["verdict"] == "INCOMPLETE"
    assert v["evaluated_cases"] == 40
    assert v["overall"].startswith("40/40")


def test_thresholds_match_legacy_absolute_rule():
    # 48/50 = 0.96 format-valid boundary: exactly at threshold passes
    rows = fifty([f"c{i}" for i in range(50)])
    for r in rows[:2]:
        r["format_valid"] = False
    v = compute_verdict(rows, 50)
    assert v["verdict"] == "PASS"
    for r in rows[:3]:
        r["format_valid"] = False
    v = compute_verdict(rows, 50)
    assert v["verdict"] == "PARTIAL"


def test_subset_omitting_category_does_not_crash():
    # only bash cases selected: other categories absent entirely
    rows = [mk(f"b{i}", "bash", True) for i in range(5)]
    v = compute_verdict(rows, 50)
    assert v["verdict"] == "INCOMPLETE"
    assert v["categories"]["workflow"] == "n/a"


def test_empty_selection_is_incomplete_not_fail():
    v = compute_verdict([], 50)
    assert v["verdict"] == "INCOMPLETE"
