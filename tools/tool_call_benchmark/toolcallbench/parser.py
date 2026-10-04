# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tool-call output parsing for the tool_call_benchmark.

Model responses arrive in one of four dialect families depending on how the
model was trained: JSON objects (Llama-3, Qwen2/3, Hermes), XML tags
(Qwen3.5), python-style parenthesised calls (LFM2.5) and gemma-style DSL.

``derive_parser_config`` probes a tokenizer's chat template and derives the
dialect parameters without any model-specific code. ``ToolCallParser`` then
parses raw model output into tool calls and a text answer.
"""

import copy
import json
import re
from dataclasses import dataclass, field


import json, re


class _DialectCore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.json_style = cfg.get("json_style", False)
        self.xml_style = cfg.get("xml_style", False)
        self.cs = cfg.get("call_start", "")
        self.ce = cfg.get("call_end", "")
        self.ao = cfg.get("arg_open", "{")
        self.ac = cfg.get("arg_close", "}")
        self.func_open = cfg.get("func_open", "<function=")
        self.func_close = cfg.get("func_close", "</function>")
        self.param_open = cfg.get("param_open", "<parameter=")
        self.param_close = cfg.get("param_close", "</parameter>")
        sd = cfg.get("str_delim")
        if isinstance(sd, list):
            self.sdl, self.sdr = (sd if len(sd) == 2 else (sd[0], sd[0]))
        else:
            self.sdl = self.sdr = sd

    # ---- block extraction -------------------------------------------------
    def blocks(self, text):
        if self.json_style:
            return self._json_blocks(text)
        if not self.cs and not self.ce:
            return []
        pat = re.escape(self.cs) + r"(.*?)" + (re.escape(self.ce) if self.ce else r"$")
        return re.findall(pat, text, re.DOTALL)

    def _json_blocks(self, text):
        """Brace-balanced JSON objects containing a name key, optionally
        gated by call_start (if set) and validated against call_end (if set).
        Robust against ']' inside argument arrays (mistral) and empty markers
        (llama)."""
        out = []
        search_from = 0
        while True:
            if self.cs:
                s = text.find(self.cs, search_from)
                if s == -1:
                    break
                b = text.find("{", s + len(self.cs))
            else:
                b = text.find("{\"name\"", search_from)
                if b == -1:
                    b = text.find("{ 'name'", search_from)
                if b == -1:
                    break
                s = b
            if b == -1:
                break
            depth, i, in_str, esc = 0, b, False, False
            end = -1
            while i < len(text):
                c = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                i += 1
            if end == -1:
                break
            out.append(text[b:end + 1])
            search_from = end + 1
        return out

    # ---- value parsing ----------------------------------------------------
    def _split(self, s, sep=","):
        parts, depth, cur, i = [], 0, [], 0
        L, R = len(self.sdl or "\x00"), len(self.sdr or "\x00")
        while i < len(s):
            if self.sdl and s.startswith(self.sdl, i):
                cur.append(self.sdl)
                j = s.find(self.sdr, i + L)
                j = j if j != -1 else len(s)
                cur.append(s[i + L:j])
                if j < len(s):
                    cur.append(self.sdr)
                i = j + R
                continue
            c = s[i]
            if c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
            if c == sep and depth == 0:
                parts.append("".join(cur))
                cur = []
            else:
                cur.append(c)
            i += 1
        parts.append("".join(cur))
        return [p for p in parts if p.strip()]

    def _value(self, v):
        v = v.strip()
        if self.sdl and v.startswith(self.sdl):
            if not (self.sdr and v.endswith(self.sdr)):
                raise ValueError("unterminated string")
            return v[len(self.sdl): len(v) - len(self.sdr)]
        if v == "true":
            return True
        if v == "false":
            return False
        if v == "null":
            return None
        if v.startswith(self.ao) and v.endswith(self.ac):
            return self._object(v[1:-1])
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            return [] if not inner else [self._value(p) for p in self._split(inner)]
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            raise ValueError(f"unparseable {v[:50]!r}")

    def _object(self, body):
        out = {}
        if not body.strip():
            return out
        for part in self._split(body):
            depth, idx, i = 0, -1, 0
            L, R = len(self.sdl or "\x00"), len(self.sdr or "\x00")
            while i < len(part):
                if self.sdl and part.startswith(self.sdl, i):
                    j = part.find(self.sdr, i + L)
                    i = (j + R) if j != -1 else len(part)
                    continue
                c = part[i]
                if c in "{[":
                    depth += 1
                elif c in "}]":
                    depth -= 1
                elif c == ":" and depth == 0:
                    idx = i
                    break
                i += 1
            sep = "=" if self.ao == "(" else ":"
            idx = part.find(sep)
            if idx == -1 and sep == "=":
                idx = part.find(":")
            if idx == -1:
                raise ValueError(f"no {sep!r} in {part[:50]!r}")
            key = part[:idx].strip()
            if self.sdl and key.startswith(self.sdl) and key.endswith(self.sdr) and len(key) >= len(self.sdl) + len(self.sdr):
                key = key[len(self.sdl): len(key) - len(self.sdr)]
            out[key] = self._value(part[idx + 1:])
        return out


    def _xml_value(self, v):
        v = v.strip()
        if v.startswith(("\"", "'")) and v.endswith(("\"", "'")) and len(v) >= 2:
            v = v[1:-1]
        if v == "true":
            return True
        if v == "false":
            return False
        if v == "null":
            return None
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        if v.startswith("[") or v.startswith("{"):
            try:
                return json.loads(v)
            except Exception:
                pass
        return v

    def _parse_xml_block(self, b):
        m = re.search(r"<function=([^>\s]+)>", b)
        if not m:
            return None
        name = m.group(1)
        args = {}
        for pm in re.finditer(r"<parameter=([^>\s]+)>([\s\S]*?)</parameter>", b):
            args[pm.group(1)] = self._xml_value(pm.group(2))
        return {"name": name, "arguments": args}

    def parse(self, raw):
        """Returns (calls, malformed) where calls=[{"name","arguments"}]."""
        calls, malformed = [], False
        if self.xml_style:
            pat = re.escape(self.cs) + r"([\s\S]*?)" + re.escape(self.ce)
            for b in re.findall(pat, raw):
                c = self._parse_xml_block(b)
                if c:
                    calls.append(c)
                else:
                    malformed = True
            return calls, malformed
        for b in self.blocks(raw):
            b = b.strip()
            if not self.json_style and self.ac:
                # lfm-style bracket wrapper: [name(args)] -> trailing ] not part of args
                while b.endswith("]") and not b.endswith(self.ac):
                    b = b[:-1].rstrip()
            if self.json_style:
                try:
                    d = json.loads(b)
                    name = d.get("name") or d.get("function", {}).get("name")
                    args = d.get("arguments") or d.get("parameters") or {}
                    if isinstance(args, str):
                        args = json.loads(args)
                    if not isinstance(args, dict):
                        malformed = True
                        continue
                    if name:
                        calls.append({"name": name, "arguments": args})
                    else:
                        malformed = True
                except Exception:
                    malformed = True
                continue
            ob = b.find(self.ao)
            if ob == -1 or not b.rstrip().endswith(self.ac):
                malformed = True
                continue
            name = b[:ob].strip()
            try:
                args = self._object(b[ob + 1: b.rstrip().rfind(self.ac)])
            except ValueError:
                malformed = True
                continue
            calls.append({"name": name, "arguments": args})
        return calls, malformed

    def _parse_calls_public(self, stripped):
        return self._parse_calls(stripped)

    def _parse_calls(self, stripped):
        calls, malformed = [], False
        # visible start markers that never form a complete block are malformed
        if self.cs and self.ce and self.cs in stripped:
            pair = re.escape(self.cs) + r"[\s\S]*?" + re.escape(self.ce)
            if len(re.findall(pair, stripped)) < stripped.count(self.cs):
                malformed = True
        if self.xml_style:
            pat = re.escape(self.cs) + r"([\s\S]*?)" + re.escape(self.ce)
            for b in re.findall(pat, stripped):
                c = self._parse_xml_block(b)
                if c:
                    calls.append(c)
                else:
                    malformed = True
            return calls, malformed
        for b in self.blocks(stripped):
            b = b.strip()
            if not self.json_style and self.ac:
                while b.endswith("]") and not b.endswith(self.ac):
                    b = b[:-1].rstrip()
            if self.json_style:
                try:
                    d = json.loads(b)
                    name = d.get("name") or d.get("function", {}).get("name")
                    args = d.get("arguments") or d.get("parameters") or {}
                    if isinstance(args, str):
                        args = json.loads(args)
                    if not isinstance(args, dict):
                        malformed = True
                        continue
                    if name:
                        calls.append({"name": name, "arguments": args})
                    else:
                        malformed = True
                except Exception:
                    malformed = True
                continue
            ob = b.find(self.ao)
            if ob == -1 or not b.rstrip().endswith(self.ac):
                malformed = True
                continue
            name = b[:ob].strip()
            try:
                args = self._object(b[ob + 1: b.rstrip().rfind(self.ac)])
            except ValueError:
                malformed = True
                continue
            calls.append({"name": name, "arguments": args})
        return calls, malformed

    # keep the prototype parse() shape unavailable; use parse()
    def parse_prototype(self, raw):
        return self._parse_calls_public(raw)


PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "probe_tool",
        "description": "probe",
        "parameters": {
            "type": "object",
            "properties": {"s": {"type": "string"}, "n": {"type": "number"}},
            "required": ["s"],
        },
    },
}


def _probe_call(name, args):
    return {"id": "abcdefghi", "type": "function",
            "function": {"name": name, "arguments": args}}


def _render_ids(tok, messages, tools, agp=False):
    out = tok.apply_chat_template(messages, tools=tools, tokenize=True,
                                  add_generation_prompt=agp)
    if not isinstance(out, list):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return out


def _variants(messages):
    """Yield content-variant copies of tool-call assistant messages.

    Templates differ in what they accept for assistant tool-call turns
    (empty string, None or absent content); try all three shapes.
    """
    yield messages
    v = copy.deepcopy(messages)
    for m in v:
        if m.get("tool_calls") and m.get("content") == "":
            m["content"] = None
    yield v
    v2 = copy.deepcopy(messages)
    for m in v2:
        if m.get("tool_calls") and "content" in m:
            del m["content"]
    yield v2


def _render_try(tok, messages, tools, agp=False):
    last = None
    for v in _variants(messages):
        try:
            return _render_ids(tok, v, tools, agp=agp)
        except Exception as e:
            last = e
    raise last


def _added_text(tok, base_ids, full_ids):
    """Common prefix/suffix strip at token level, decode the added span."""
    n = min(len(base_ids), len(full_ids))
    pre = 0
    while pre < n and base_ids[pre] == full_ids[pre]:
        pre += 1
    suf = 0
    while suf < n - pre and base_ids[len(base_ids) - 1 - suf] == full_ids[len(full_ids) - 1 - suf]:
        suf += 1
    added = full_ids[pre: len(full_ids) - suf]
    return tok.decode(added, skip_special_tokens=False), added


def derive_parser_config(tok):
    """Derive the response dialect config from a tokenizer's chat template.

    :param tok: HF tokenizer carrying the model's chat template.
    :return: dict of dialect parameters, or a dict with an ``error`` key.
    """
    tools = [PROBE_TOOL]
    u = [{"role": "user", "content": "hi"}]
    plain = u + [{"role": "assistant", "content": "ok"}]
    p0 = _render_try(tok, plain, tools)
    p1 = _render_try(tok, u + [{"role": "assistant", "content": "",
                                "tool_calls": [_probe_call("probe_tool", {"s": "STRVAL"})]}], tools)
    p2 = _render_try(tok, u + [{"role": "assistant", "content": "",
                                "tool_calls": [_probe_call("probe_tool", {"n": 42})]}], tools)
    try:
        p3 = _render_try(tok, u + [{"role": "assistant", "content": "",
                                    "tool_calls": [_probe_call("probe_tool", {"s": "A"}), _probe_call("probe_tool", {"s": "B"})]}], tools)
    except Exception:
        # some templates (llama3) only support one call per message; use two turns
        try:
            p3 = _render_try(tok, u + [{"role": "assistant", "content": "",
                                        "tool_calls": [_probe_call("probe_tool", {"s": "A"})]},
                                       {"role": "user", "content": "and"},
                                       {"role": "assistant", "content": "",
                                        "tool_calls": [_probe_call("probe_tool", {"s": "B"})]}], tools)
        except Exception:
            p3 = p1
    p4 = _render_try(tok, u + [{"role": "assistant", "content": "",
                                "tool_calls": [_probe_call("probe_tool", {"s": "X"})]},
                               {"role": "tool", "tool_call_id": "abcdefghi", "content": "RESPBODY"}], tools, agp=True)

    cfg = {"unknown_fields": [], "json_style": False}
    try:
        t1, _ = _added_text(tok, p0, p1)
        t2, _ = _added_text(tok, p0, p2)
        t3, _ = _added_text(tok, p0, p3)
        t4, _ = _added_text(tok, p1, p4)
    except Exception as e:
        cfg["error"] = f"render failed: {e}"
        return cfg

    cfg["raw_examples"] = {"p1": t1[:300], "p2": t2[:200], "p3": t3[:400], "p4": t4[:300]}

    # --- detect XML style (qwen3.5 family): <function=NAME>...<parameter=K>V</parameter>
    if "<function=" in t1 and "<parameter=" in t1:
        cfg["xml_style"] = True
        mcs = re.search(r"([\s\S]*?)<function=", t1)
        pre = mcs.group(1) if mcs else ""
        mpre = re.search(r"([^\n]+)$", pre.rstrip("\n") if not pre.strip() else pre)
        cfg["call_start"] = (mpre.group(1).strip() if mpre else "") or "<tool_call>"
        mend = re.search(r"</function>([\s\S]*?)$", t1)
        post = mend.group(1) if mend else ""
        mpost = re.search(r"^\s*([^\n]+)", post.strip())
        cfg["call_end"] = (mpost.group(1).strip() if mpost else "") or "</tool_call>"
        cfg["func_open"], cfg["func_close"] = "<function=", "</function>"
        cfg["param_open"], cfg["param_close"] = "<parameter=", "</parameter>"
        cfg["str_delim"] = None
        cfg["number_bare"] = True
        return cfg

    # --- detect JSON style (llama3/qwen3/hermes family): {"name": "...", "arguments"/"parameters": {...}}
    if re.search(r"[\"']name[\"']\s*:", t1):
        cfg["json_style"] = True
        brace = t1.index("{")
        pre = t1[:brace]
        mpre = re.search(r"([^\n]+)$", pre.rstrip("\n") if not pre.strip() else pre)
        cand = (mpre.group(1) if mpre else "").strip()
        cfg["call_start"] = cand
        tail = t1.rstrip()
        if "}" in tail:
            after = tail[tail.rindex("}") + 1:]
            m2 = re.search(r"(\S+)", after)
            cfg["call_end"] = m2.group(1) if m2 else ""
        else:
            cfg["call_end"] = ""
        cfg["arg_open"], cfg["arg_close"] = "{", "}"
        cfg["str_delim"] = '"'
        return cfg

    # --- custom DSL style (gemma4 etc.): find STRVAL, extract exact surroundings
    idx = t1.find("STRVAL")
    if idx == -1:
        cfg["error"] = "STRVAL not found in rendered probe; unsupported template"
        return cfg
    left, right = t1[:idx], t1[idx + len("STRVAL"):]

    # string delimiters (exact runs of wrapper chars adjacent to the value)
    lm = re.search(r"([\"'<>|«»`]+)$", left)
    rm = re.match(r"^([\"'<>|«»`]+)", right)
    ldelim, rdelim = (lm.group(1) if lm else ""), (rm.group(1) if rm else "")
    if ldelim and ldelim == rdelim:
        cfg["str_delim"] = ldelim
    elif ldelim or rdelim:
        cfg["str_delim"] = [ldelim, rdelim]
    else:
        cfg["str_delim"] = None
        cfg["unknown_fields"].append("str_delim")

    # arg block opener/closer
    mo = re.search(r"([{(])[^{(]*$", left)
    cfg["arg_open"] = mo.group(1) if mo else None
    if not mo:
        cfg["unknown_fields"].append("arg_open")
        return cfg
    pre = left[: mo.start()]
    mname = re.search(r"([A-Za-z0-9_.\-]+)\s*$", pre)
    cfg["name_example"] = mname.group(1) if mname else None
    cfg["call_start"] = pre[: mname.start()].strip("\n") if mname else pre.strip("\n")

    body_right = right[len(rdelim):] if rdelim else right
    mc = re.match(r"\s*([}\)])", body_right)
    cfg["arg_close"] = mc.group(1) if mc else None
    if mc:
        rest = body_right[mc.end():]
        # lfm-style: ]<|tool_call_end|> (bracket wrapper before the marker)
        mend = re.match(r"\s*[\]\)\s]*(<[^>]*>|\|>|>)", rest)
        cfg["call_end"] = mend.group(1) if mend else ""

    # number syntax
    cfg["number_bare"] = bool(re.search(r"[\s{,(:]42[}\s,)]", t2)) and "STRVAL" not in t2

    # call separator: text between end of first call and start of second in t3
    msep = re.search(r"\}\s*(.*?)<", t3.replace("STRVAL", ""))
    first_a = t3.find("A<") if "A" in t3 else -1
    sep = None
    if cfg.get("call_start") and cfg.get("call_end"):
        start2 = t3.find(cfg["call_start"], 1)
        end1 = t3.find(cfg["call_end"], 1)
        if start2 != -1 and end1 != -1 and end1 < start2:
            sep = t3[end1 + len(cfg["call_end"]): start2]
    cfg["call_sep"] = sep

    # response block (exact, from p4 diff vs p1)
    cfg["response_block"] = t4.strip() if t4.strip() else None
    return cfg


@dataclass
class ParsedOutput:
    """Parsed model output for one turn."""

    calls: list = field(default_factory=list)
    text: str = ""
    error: str = None  # None | "malformed" | "call_in_thought"


class ToolCallParser(_DialectCore):
    """Config-driven parser with think-block handling for all dialects."""

    def parse(self, raw):
        """Parse one raw model turn.

        Strips reasoning blocks (gemma channel / qwen think), flags calls
        hidden inside them, parses visible calls in the model's dialect.

        :param raw: raw generated text for one turn.
        :return: ParsedOutput with calls, cleaned text and error status.
        """
        stripped = re.sub(r"<\|channel>thought\n?.*?<channel\|>", "", raw, flags=re.DOTALL)
        think_removed = re.sub(r"<think>[\s\S]*?</think>", "", stripped, flags=re.DOTALL)
        had_call_in_thought = False
        for blk in re.finditer(r"<think>([\s\S]*?)</think>", stripped):
            inner = blk.group(1)
            if "<tool_call>" in inner or "<function=" in inner or (self.cs and self.cs in inner):
                had_call_in_thought = True
        stripped = think_removed
        if not had_call_in_thought:
            had_call_in_thought = (("<|tool_call>" in raw) or (self.cs and self.cs in raw)
                                   or "<function=" in raw) \
                and not (self.cs and self.cs in stripped) and "<function=" not in stripped
        calls, malformed = self._parse_calls_public(stripped)
        error = None
        if had_call_in_thought:
            error = "call_in_thought"
        elif malformed:
            error = "malformed"
        return ParsedOutput(calls=calls, text=stripped, error=error)
