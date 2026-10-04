# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Parser regression across the four tool-call dialect families."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolcallbench.parser import ToolCallParser


def gemma_parser():
    return ToolCallParser({
        "json_style": False, "call_start": "<|tool_call|>", "call_end": "<|end|>",
        "arg_open": "{", "arg_close": "}", "str_delim": ['<|"|>', '<|"|>'],
    })


def json_parser():
    return ToolCallParser({"json_style": True, "call_start": "", "call_end": ""})


def xml_parser():
    return ToolCallParser({"xml_style": True, "call_start": "<tool_call>",
                           "call_end": "</tool_call>",
                           "func_open": "<function=", "param_open": "<parameter="})


def lfm_parser():
    return ToolCallParser({
        "json_style": False, "call_start": "<|tool_call_start|>[",
        "call_end": "<|tool_call_end|>",
        "arg_open": "(", "arg_close": ")", "str_delim": "'", "call_sep": None,
    })


def test_gemma_dsl():
    calls = gemma_parser().parse(
        'thought here<|tool_call|>call:get_weather{city:<|"|>Paris<|"|>,'
        'days:<|"|>3<|"|>}<|end|>')
    assert calls.error is None
    assert calls.calls == [{"name": "call:get_weather",
                            "arguments": {"city": "Paris", "days": "3"}}]


def test_json_family():
    calls = json_parser().parse(
        'bla bla {"name": "get_weather", "parameters": {"city": "Paris", "days": 3}} trailing')
    assert calls.error is None
    assert calls.calls == [{"name": "get_weather",
                            "arguments": {"city": "Paris", "days": 3}}]


def test_xml_family():
    calls = xml_parser().parse(
        "<think>reasoning</think><tool_call><function=run_command>"
        "<parameter=command>pytest tests/test_io.py -q</parameter></function></tool_call>")
    assert calls.error is None
    assert calls.calls == [{"name": "run_command",
                            "arguments": {"command": "pytest tests/test_io.py -q"}}]


def test_lfm_paren_style():
    calls = lfm_parser().parse(
        "reasoning</think><|tool_call_start|>[run_command(command='git checkout -b feat/x')]"
        "<|tool_call_end|><|im_end|>")
    assert calls.error is None
    assert calls.calls == [{"name": "run_command",
                            "arguments": {"command": "git checkout -b feat/x"}}]


def test_lfm_typed_args():
    calls = lfm_parser().parse(
        "<|tool_call_start|>[scale(replicas=4, cluster='prod')]<|tool_call_end|>")
    assert calls.error is None
    assert calls.calls[0]["arguments"]["replicas"] == 4


def test_gemma_unaffected_by_paren_fixes():
    calls = gemma_parser().parse('<|tool_call|>call:f{a:<|"|>1<|"|>}<|end|>')
    assert calls.error is None
    assert calls.calls[0]["arguments"] == {"a": "1"}


def test_call_in_thought_flagged():
    calls = xml_parser().parse(
        "<think>let me call <tool_call><function=x><parameter=p>1</parameter>"
        "</function></tool_call></think> no")
    assert calls.error == "call_in_thought"


def test_orphan_start_marker_is_malformed():
    # <tool_call><function=x> with no end marker must not count as format-valid
    calls = xml_parser().parse("<tool_call><function=x><parameter=p>1</parameter>")
    assert calls.error == "malformed"
    assert calls.calls == []


def test_orphan_lfm_marker_is_malformed():
    calls = lfm_parser().parse("<|tool_call_start|>[run_command(command='x")
    assert calls.error == "malformed"


def test_ce_empty_dialect_does_not_crash():
    calls = ToolCallParser({"json_style": True, "call_start": "[TOOL]",
                            "call_end": ""})
    out = calls.parse('[TOOL] {"name":"f","arguments":{}}')
    assert out.error is None and out.calls[0]["name"] == "f"


def test_non_object_arguments_are_malformed():
    calls = ToolCallParser({"json_style": True, "call_start": "",
                            "call_end": ""})
    out = calls.parse('{"name":"run_command","arguments":["x"]}')
    assert out.error == "malformed" and out.calls == []


def test_xml_block_requires_closed_tags():
    calls = ToolCallParser({"xml_style": True, "call_start": "<tool_call>",
                            "call_end": "</tool_call>"})
    a = calls.parse("<tool_call><function=f><parameter=p>1</parameter></function></tool_call>")
    assert a.calls and a.error is None
    b = calls.parse("<tool_call><function=f><parameter=p>1</parameter>")
    assert b.error == "malformed" and b.calls == []
    c = calls.parse("<tool_call><function=f><parameter=p>1</function></tool_call>")
    assert c.error == "malformed"
