# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hostile-input tests: model output is untrusted. Type-garbage values for
every argument of every tool across case kinds must grade deterministically
and never raise."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from toolcallbench.dataset import load_dataset
from toolcallbench.engine import CaseEngine

BAD_VALUES = [[], {}, None, True, 3.5, ("a",)]

TOOLS_ARGS = {
    "run_command": ["command"],
    "read_file": ["path"],
    "write_file": ["path", "content"],
    "edit_file": ["path", "old_str", "new_str"],
    "search": ["pattern", "path"],
    "list_dir": ["path"],
}


def _cases_by_kind(dataset):
    meta, data = dataset
    cases = list(data.values()) if isinstance(data, dict) else data
    seen = {}
    for c in cases:
        seen.setdefault(c["kind"], c)
    return meta, seen


def test_hostile_args_never_raise(dataset):
    meta, by_kind = _cases_by_kind(dataset)
    kinds = [k for k in by_kind if k != "catalog"] or list(by_kind)[:1]
    for kind in kinds:
        case = by_kind[kind]
        for tool, fields in TOOLS_ARGS.items():
            for field in fields:
                for bad in BAD_VALUES:
                    args = {f: "ok" for f in fields}
                    args[field] = bad
                    eng = CaseEngine(case, meta)
                    eng.respond([{"name": tool, "arguments": args}], text="")
                    g = eng.grade()  # must not raise
                    assert isinstance(g, dict) and "correct" in g


def test_hostile_call_shapes_never_raise(dataset):
    meta, by_kind = _cases_by_kind(dataset)
    case = next(iter(by_kind.values()))
    for shape in [None, [], {}, {"name": None, "arguments": {}},
                  {"name": "run_command"}, {"name": "run_command", "arguments": None},
                  {"name": 42, "arguments": {"command": "ls"}},
                  {"name": "run_command", "arguments": {"command": None}}]:
        eng = CaseEngine(case, meta)
        eng.respond([shape], text="")
        g = eng.grade()
        assert isinstance(g, dict)


def test_hostile_commands_never_raise(dataset):
    meta, by_kind = _cases_by_kind(dataset)
    case = next(iter(by_kind.values()))
    for cmd in ["", "   ", "&&", "|||", "|&;", "cat <<EOF", "$(rm -rf /)",
                "`x`", "cmd $((", "'", '"', "git", "-", "\x00", "a" * 5000,
                "git status&&rm -rf /", "ls & rm -rf /", "cat x|kubectl apply -f -"]:
        eng = CaseEngine(case, meta)
        eng.respond([{"name": "run_command", "arguments": {"command": cmd}}], text="")
        eng.grade()
