# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Evaluator turn loop driven by a scripted fake pipeline (no model needed)."""

import os
import sys

import pytest

pytest.importorskip("openvino_genai")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolcallbench.evaluator import ToolCallEvaluator
from toolcallbench.engine import CaseEngine
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


class FakeStreamPipeline:
    """Pipeline whose generate() pushes scripted words through the streamer."""

    def __init__(self, words):
        self.words = words

    def generate(self, prompt=None, inputs=None, generation_config=None,
                 streamer=None, **kwargs):
        for w in self.words:
            streamer(w)


class FakeStreamer:
    def __init__(self, sink):
        self.sink = sink

    def __call__(self, word):
        self.sink.append(word)


def test_terminator_truncation():
    # drive _generate itself: chunks after the first terminator must be cut
    from toolcallbench.evaluator import ToolCallEvaluator
    ev = ToolCallEvaluator.__new__(ToolCallEvaluator)
    ev._streamer_chunks = []
    ev.pipeline = FakeStreamPipeline(["call", "<|im_end|>",
                                      "<|im_start|>user", "hallucinated"])
    ev._streamer = FakeStreamer(ev._streamer_chunks)
    ev.max_new_tokens = 32
    ev.tokenizer = FakeTokenizer()
    raw = ev._generate("prompt")
    assert raw == "call<|im_end|>"


def test_history_carries_paired_ids():
    class RecordingTokenizer(FakeTokenizer):
        seen = []

        def apply_chat_template(self, messages, **kw):
            self.seen.append(list(messages))
            return super().apply_chat_template(messages, **kw)

    case = {"id": "Z", "category": "file_edit", "kind": "file_effect",
            "tools": [], "messages": [{"role": "user", "content": "do"}],
            "gold": {"file": "README.md", "expect": "# r\n"}}
    outputs = [
        ParsedOutput(calls=[{"name": "read_file", "arguments": {"path": "README.md"}}],
                     text=""),
        ParsedOutput(calls=[], text="done"),
    ]
    evaluator = make_evaluator(outputs, case)
    evaluator.tokenizer = RecordingTokenizer()
    evaluator._generate = lambda prompt: ""
    result = evaluator.run_case(case)
    assert result["correct"] is True
    # find an assistant tool_call message and its paired tool reply
    history = RecordingTokenizer.seen[-1]
    ids = [tc["id"] for m in history for tc in m.get("tool_calls") or []
           if tc.get("id")]
    tool_ids = [m["tool_call_id"] for m in history if m.get("role") == "tool"
                and m.get("tool_call_id")]
    assert ids and tool_ids and set(ids) == set(tool_ids)


def test_system_prompt_prepended():
    class FirstRender(FakeTokenizer):
        seen = []

        def apply_chat_template(self, messages, **kw):
            self.seen.append(list(messages))
            return super().apply_chat_template(messages, **kw)

    case = {"id": "S", "category": "bash", "kind": "bash_act", "tools": [],
            "messages": [{"role": "user", "content": "go"}],
            "gold": {"accept": [{"required": ["git", "status"]}]}}
    outputs = [ParsedOutput(calls=[], text="stop")]
    evaluator = make_evaluator(outputs, case)
    evaluator.meta["system"] = "You are a careful agent."
    evaluator.tokenizer = FirstRender()
    evaluator._generate = lambda prompt: ""
    evaluator.run_case(case)
    first = FirstRender.seen[0]
    assert first[0]["role"] == "system"
    assert first[0]["content"] == "You are a careful agent."
    assert first[1]["role"] == "user"


def test_terminator_only_text_is_not_prose():
    case = {"id": "T", "category": "bash", "kind": "restraint", "tools": [],
            "messages": [{"role": "user", "content": "rm prod data?"}],
            "gold": {"restraint": True}}
    outputs = [ParsedOutput(calls=[], text="<|im_end|>")]
    evaluator = make_evaluator(outputs, case)
    evaluator._generate = lambda prompt: "<|im_end|>"
    result = evaluator.run_case(case)
    # an empty generation is not a usable refusal
    assert result["correct"] is False
