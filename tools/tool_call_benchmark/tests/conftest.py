# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolcallbench.dataset import load_dataset
from toolcallbench.engine import CaseEngine


@pytest.fixture(scope="session")
def dataset():
    meta, cases = load_dataset()
    return meta, {c["id"]: c for c in cases}


@pytest.fixture(scope="session")
def meta(dataset):
    return dataset[0]


@pytest.fixture(scope="session")
def cases(dataset):
    return dataset[1]


@pytest.fixture(scope="session")
def repo(meta):
    return meta["repo_files"]


def rc(cmd):
    return [{"name": "run_command", "arguments": {"command": cmd}}]


def rf(path):
    return [{"name": "read_file", "arguments": {"path": path}}]


def edit(path, old, new):
    return [{"name": "edit_file", "arguments": {"path": path, "old_str": old, "new_str": new}}]


def writef(path, content):
    return [{"name": "write_file", "arguments": {"path": path, "content": content}}]


def run_case_transcript(case, meta, turns):
    """Replay a list of per-turn call-lists through a fresh engine."""
    engine = CaseEngine(case, meta)
    for calls in turns:
        if engine.done:
            break
        # a no-call turn means the model answered in prose
        engine.respond(calls, text="here is what I found" if not calls else "")
    return engine.grade()
