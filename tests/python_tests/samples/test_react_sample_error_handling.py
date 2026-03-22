# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import requests
import pytest

from conftest import SAMPLES_PY_DIR


@pytest.fixture(scope="module")
def react_module():
    module_path = SAMPLES_PY_DIR / "text_generation/react_sample.py"
    spec = importlib.util.spec_from_file_location("react_sample", module_path)
    assert spec is not None, f"Failed to load module spec from {module_path}"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.samples
@pytest.mark.agent
@pytest.mark.llm
def test_call_tool_handles_invalid_json_args(react_module):
    result = react_module.call_tool("get_weather", "not-json")
    payload = json.loads(result)
    assert payload["tool"] == "get_weather"
    assert "Failed to parse tool arguments" in payload["error"]


@pytest.mark.samples
@pytest.mark.agent
@pytest.mark.llm
def test_call_tool_handles_missing_required_argument(react_module):
    result = react_module.call_tool("get_weather", "{}")
    payload = json.loads(result)
    assert payload["tool"] == "get_weather"
    assert "Missing or invalid required argument: city_name" == payload["error"]


@pytest.mark.samples
@pytest.mark.agent
@pytest.mark.llm
def test_call_tool_handles_unsupported_tool(react_module):
    result = react_module.call_tool("unknown_tool", "{}")
    payload = json.loads(result)
    assert payload["tool"] == "unknown_tool"
    assert payload["error"] == "Unsupported tool"


@pytest.mark.samples
@pytest.mark.agent
@pytest.mark.llm
def test_call_tool_handles_weather_request_exception(monkeypatch, react_module):
    def _raise_request_exception(*args, **kwargs):
        raise requests.RequestException("network unavailable")

    monkeypatch.setattr(react_module.requests, "get", _raise_request_exception)
    result = react_module.call_tool("get_weather", '{"city_name":"London"}')
    payload = json.loads(result)
    assert payload["tool"] == "get_weather"
    assert "Weather request failed" in payload["error"]


class _FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True):
        return "PROMPT\n"


class _FakeLLMPipeline:
    def __init__(self):
        self._outputs = [
            "\nAction: get_weather\nAction Input: not-json\nObservation:",
            "Final Answer: fallback response",
        ]

    def get_tokenizer(self):
        return _FakeTokenizer()

    def get_generation_config(self):
        return object()

    def generate(self, prompt, config, streamer):
        if self._outputs:
            return self._outputs.pop(0)
        return "Final Answer: done"


@pytest.mark.samples
@pytest.mark.agent
@pytest.mark.llm
def test_llm_with_tool_continues_after_tool_error(react_module):
    fake_pipe = _FakeLLMPipeline()
    text, history = react_module.llm_with_tool(
        fake_pipe,
        prompt="question",
        history=[],
        list_of_tool_info=react_module.tools,
        max_steps=4,
    )

    assert "Failed to parse tool arguments" in text
    assert "Final Answer: fallback response" in text
    assert len(history) == 1
