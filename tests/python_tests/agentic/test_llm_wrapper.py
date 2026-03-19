# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


class _FakeGenerationConfig:
    def __init__(self):
        self.max_new_tokens = 0
        self.temperature = 0.0
        self.top_p = 1.0
        self.num_return_sequences = 1
        self.repetition_penalty = 1.0
        self.stop_strings = set()
        self.validated = False

    def validate(self):
        self.validated = True


class _FakeDecodedResults:
    def __init__(self, texts):
        self.texts = texts


class _FakeLLMPipeline:
    instances = []
    stream_chunks = ["alpha", " ", "beta"]
    output_text = "wrapped response"
    should_raise = False

    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.last_inputs = None
        self.last_config = None
        self.last_streamer = None
        _FakeLLMPipeline.instances.append(self)

    def get_generation_config(self):
        return _FakeGenerationConfig()

    def generate(self, inputs, config, streamer=None):
        self.last_inputs = inputs
        self.last_config = config
        self.last_streamer = streamer

        if _FakeLLMPipeline.should_raise:
            raise RuntimeError("generation failed")

        if streamer is not None:
            for chunk in _FakeLLMPipeline.stream_chunks:
                streamer(chunk)
            return _FakeDecodedResults(["".join(_FakeLLMPipeline.stream_chunks)])

        return _FakeDecodedResults([_FakeLLMPipeline.output_text])


@pytest.fixture(scope="module")
def wrapper_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "samples/python/agentic/llm_wrapper.py"
    spec = importlib.util.spec_from_file_location("test_llm_wrapper_module", module_path)
    assert spec is not None, f"Failed to create module spec for {module_path}"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_module():
    repo_root = Path(__file__).resolve().parents[3]
    agentic_dir = repo_root / "samples/python/agentic"
    module_path = repo_root / "samples/python/agentic/example_agent.py"
    spec = importlib.util.spec_from_file_location("test_example_agent_module", module_path)
    assert spec is not None, f"Failed to create module spec for {module_path}"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(agentic_dir))
    spec.loader.exec_module(module)
    try:
        sys.path.remove(str(agentic_dir))
    except ValueError:
        pass
    return module


@pytest.fixture(autouse=True)
def patch_pipeline(monkeypatch, wrapper_module):
    _FakeLLMPipeline.instances.clear()
    _FakeLLMPipeline.should_raise = False
    monkeypatch.setattr(wrapper_module.ov_genai, "LLMPipeline", _FakeLLMPipeline)


@pytest.fixture
def model_instance(tmp_path, wrapper_module):
    model_path = tmp_path / "fake_model"
    model_path.mkdir(parents=True, exist_ok=True)
    return wrapper_module.OpenVINOChatModel(model_path=str(model_path), device="CPU")


@pytest.mark.samples
@pytest.mark.agent
def test_init_rejects_missing_model_path(wrapper_module):
    with pytest.raises(ValueError, match="Model path does not exist"):
        wrapper_module.OpenVINOChatModel(model_path="missing/path")


@pytest.mark.samples
@pytest.mark.agent
def test_init_rejects_multi_sequence_default(tmp_path, wrapper_module):
    model_path = tmp_path / "fake_model"
    model_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="num_return_sequences=1"):
        wrapper_module.OpenVINOChatModel(model_path=str(model_path), num_return_sequences=2)


@pytest.mark.samples
@pytest.mark.agent
def test_validate_kwargs_rejects_unsupported_key(model_instance):
    with pytest.raises(ValueError, match="Unsupported generation kwargs"):
        model_instance._validate_kwargs({"bad_key": 1})


@pytest.mark.samples
@pytest.mark.agent
def test_validate_kwargs_rejects_runtime_multi_sequence(model_instance):
    with pytest.raises(ValueError, match="num_return_sequences=1"):
        model_instance._validate_kwargs({"num_return_sequences": 2})


@pytest.mark.samples
@pytest.mark.agent
def test_validate_kwargs_rejects_wrong_structured_output_type(model_instance):
    with pytest.raises(TypeError, match="structured_output_config"):
        model_instance._validate_kwargs({"structured_output_config": object()})


@pytest.mark.samples
@pytest.mark.agent
def test_chat_history_conversion_maps_supported_roles(model_instance):
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="user"),
        AIMessage(content="assistant"),
    ]

    history = model_instance._to_chat_history(messages)
    assert history.get_messages() == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
        {"role": "assistant", "content": "assistant"},
    ]


@pytest.mark.samples
@pytest.mark.agent
def test_chat_history_conversion_rejects_unsupported_role(model_instance):
    messages = [ToolMessage(content="tool output", tool_call_id="tool-1")]

    with pytest.raises(ValueError, match="Unsupported message type"):
        model_instance._to_chat_history(messages)


@pytest.mark.samples
@pytest.mark.agent
def test_generate_returns_chat_result_and_applies_stop(model_instance):
    result = model_instance._generate(
        [HumanMessage(content="hello")],
        stop=["STOP"],
        max_new_tokens=42,
    )

    pipeline = _FakeLLMPipeline.instances[-1]
    assert result.generations[0].message.content == "wrapped response"
    assert pipeline.last_config.max_new_tokens == 42
    assert pipeline.last_config.stop_strings == {"STOP"}
    assert pipeline.last_config.validated is True


@pytest.mark.samples
@pytest.mark.agent
def test_stream_yields_chunks_in_order(model_instance):
    chunks = [chunk.message.content for chunk in model_instance._stream([HumanMessage(content="stream")])]
    assert chunks == _FakeLLMPipeline.stream_chunks


@pytest.mark.samples
@pytest.mark.agent
def test_stream_raises_when_generation_fails(model_instance):
    _FakeLLMPipeline.should_raise = True

    with pytest.raises(RuntimeError, match="generation failed"):
        list(model_instance._stream([HumanMessage(content="stream")]))


@pytest.mark.samples
@pytest.mark.agent
def test_safe_calculator_supports_basic_arithmetic(example_module):
    assert example_module.safe_calculator("25 * 4 + 10") == "110"
    assert example_module.safe_calculator("(3 + 5) / 2") == "4"


@pytest.mark.samples
@pytest.mark.agent
def test_safe_calculator_rejects_unsupported_expression(example_module):
    with pytest.raises(ValueError, match="Unsupported"):
        example_module.safe_calculator("abs(-1)")
