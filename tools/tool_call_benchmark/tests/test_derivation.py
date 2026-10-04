# -*- coding: utf-8 -*-
"""Derivation tests: stub templates for the four supported families."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolcallbench.parser import ToolCallParser, derive_parser_config


class StubTok:
    """apply_chat_template renders the probe tool call in a fixed dialect."""

    def __init__(self, render):
        self._render = render

    def decode(self, ids, **kw):
        return "".join(chr(i) for i in ids if 0 <= i < 0x110000)

    def apply_chat_template(self, messages, tools=None, tokenize=False,
                            add_generation_prompt=False, **kw):
        if tokenize:
            text = self.apply_chat_template(messages, tools=tools,
                                            add_generation_prompt=add_generation_prompt)
            return [ord(c) % 100000 for c in text]  # pseudo ids
        out = []
        for m in messages:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    f = tc["function"]
                    out.append(self._render(f["name"], f["arguments"]))
            else:
                out.append(m.get("role", "user") + ": " + str(m.get("content", "")))
        return "\n".join(out)


def json_render(name, args):
    import json
    return "<|start_header_id|>assistant<|end_header_id|>\n\n" + \
        json.dumps({"name": name, "parameters": args})


def xml_render(name, args):
    parts = [f"<function={name}>"]
    for k, v in args.items():
        parts.append(f"<parameter={k}>{v}</parameter>")
    parts.append("</function>")
    return "<tool_call>" + "".join(parts) + "</tool_call>"


Q = chr(39)


def py_render(name, args):
    items = ", ".join(f"{k}={Q}{v}{Q}" if isinstance(v, str) else f"{k}={v}"
                      for k, v in args.items())
    return "<|tool_call_start|>[" + name + "(" + items + ")]<|tool_call_end|>"


def dsl_render(name, args):
    # gemma style tool_code fence
    items = ", ".join(f"{k}={Q}{v}{Q}" if isinstance(v, str) else f"{k}={v}"
                      for k, v in args.items())
    return "```tool_code\n" + name + "(" + items + ")\n```"


def _parse_with(tok, sample):
    cfg = derive_parser_config(tok)
    assert "error" not in cfg, cfg
    return ToolCallParser(cfg).parse(sample)


def test_derive_json_family():
    out = _parse_with(StubTok(json_render),
                      '{"name": "run_command", "parameters": {"command": "ls"}}')
    assert out.calls[0]["name"] == "run_command"
    assert out.error is None


def test_derive_xml_family():
    out = _parse_with(StubTok(xml_render),
                      "<tool_call><function=run_command>"
                      "<parameter=command>ls</parameter></function></tool_call>")
    assert out.calls[0]["arguments"]["command"] == "ls"


def test_derive_python_call_family():
    sample = "<|tool_call_start|>[run_command(command=" + chr(39) + "ls" + chr(39) + ")]<|tool_call_end|>"
    out = _parse_with(StubTok(py_render), sample)
    assert out.calls[0]["arguments"]["command"] == "ls"


def test_derive_dsl_family():
    sample = "```tool_code\nrun_command(command=" + Q + "ls" + Q + ")\n```"
    out = _parse_with(StubTok(dsl_render), sample)
    assert out.calls and out.calls[0]["arguments"]["command"] == "ls"
