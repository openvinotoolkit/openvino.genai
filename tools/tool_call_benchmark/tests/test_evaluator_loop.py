# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Evaluator turn loop driven by a scripted fake pipeline (no model needed)."""

import os
import sys

import pytest

pytest.importorskip("openvino_genai")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolcallbench.evaluator import ToolCallEvaluator
from toolcallbench.parser import ParsedOutput, ToolCallParser


class FakeTokenizer:
    """Renders minimal messages; never touches transformers."""

    bos_token = None

    def apply_chat_template(self, messages, tools=None, tokenize=False,
                            add_generation_prompt=False):
        parts = []
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content") or ""
            parts.append(f"[{role}]{content}")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                parts.append(f"[{role}]{fn.get('name')}({fn.get('arguments')})")
        return "|".join(parts)


class StubParser(ToolCallParser):
    """Returns scripted ParsedOutputs regardless of config."""

    def __init__(self, outputs):
        super().__init__({"json_style": True})
        self.outputs = list(outputs)

    def parse(self, raw):
        return self.outputs.pop(0)


def make_evaluator(raw_outputs, case):
    meta = {"system": "s", "repo_files": {"README.md": "# r\n"},
            "readonly_outputs": {}, "version": "coding_agent_v1"}
    evaluator = ToolCallEvaluator.__new__(ToolCallEvaluator)
    evaluator.pipeline = None
    evaluator.tokenizer = FakeTokenizer()
    evaluator.max_new_tokens = 64
    evaluator.meta = meta
    evaluator.cases = [case]
    evaluator.parser = StubParser(raw_outputs)
    return evaluator


def test_history_one_call_per_assistant_message():
    case = {"id": "X", "category": "file_edit", "kind": "file_effect",
            "tools": [], "messages": [{"role": "user", "content": "do"}],
            "gold": {"file": "README.md", "expect": "# r\n"}}
    outputs = [
        ParsedOutput(calls=[{"name": "read_file", "arguments": {"path": "README.md"}}],
                     text=""),
        ParsedOutput(calls=[], text="done"),
    ]
    evaluator = make_evaluator(outputs, case)
    evaluator._generate = lambda prompt: ""  # responses come from the StubParser
    result = evaluator.run_case(case)
    assert result["correct"] is True


def test_terminator_truncation():
    case = {"id": "Y", "category": "bash", "kind": "bash_act",
            "tools": [], "messages": [{"role": "user", "content": "go"}],
            "gold": {"accept": [{"required": ["git", "status"]}]}}
    outputs = [
        ParsedOutput(calls=[{"name": "run_command",
                             "arguments": {"command": "git status"}}], text=""),
    ]
    evaluator = make_evaluator(outputs, case)
    # simulated hallucinated continuation past the terminator
    evaluator._generate = lambda prompt: "x"
    result = evaluator.run_case(case)
    # the StubParser consumed the scripted call; loop must finish cleanly
    assert result["turns"] == 1
