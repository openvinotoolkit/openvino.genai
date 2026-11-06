# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import pytest
from openvino_genai import (
    Tokenizer,
    VLLMParserWrapper,
    TextParserStreamer,
)
import json


def compare_dicts(dict1, dict2, skip_keys: Optional[list[str]] = None) -> bool:
    """
    Helper function to compare two dictionaries, with an option to skip certain keys.
    Returns True if the dictionaries are equivalent (considering the skip_keys), False otherwise.
    """
    if dict1.keys() != dict2.keys():
        return False

    for key in dict1:
        if skip_keys and key in skip_keys:
            continue
        if type(dict1[key]) is not type(dict2[key]):
            return False

        if isinstance(dict1[key], dict):
            if not compare_dicts(dict1[key], dict2[key], skip_keys):
                return False
        elif isinstance(dict1[key], list):
            for item1, item2 in zip(dict1[key], dict2[key]):
                if not compare_dicts(item1, item2, skip_keys):
                    return False
        else:
            if dict1[key] != dict2[key]:
                return False

    return True


def test_final_parser_llama_32_json():
    model_output = 'Here is the result: {"name": "getOpenIncidentsTool", "parameters": {}} Would you like to know more?'
    try:
        from vllm.entrypoints.openai.tool_parsers.llama_tool_parser import Llama3JsonToolParser
    except ImportError:
        pytest.skip("No vLLM package in the environment")

    model_cached = snapshot_download("gpt2")  # required to avoid HF rate limits
    parser = Llama3JsonToolParser(AutoTokenizer.from_pretrained(model_cached))
    res_vllm = parser.extract_tool_calls(model_output, None).model_dump_json()

    wrapper = VLLMParserWrapper(parser)
    message = {"content": model_output}
    wrapper.parse(message)  # modifies message in place
    assert compare_dicts(message, json.loads(res_vllm), skip_keys=["id"])


def test_final_parser_deepseek():
    model_output = "This is a reasoning section</think>This is the rest"
    try:
        from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
    except ImportError:
        pytest.skip("No vLLM package in the environment")

    model_cached = snapshot_download("deepseek-ai/DeepSeek-V3.1")  # required to avoid HF rate limits
    parser = DeepSeekR1ReasoningParser(AutoTokenizer.from_pretrained(model_cached))
    reasoning, content = parser.extract_reasoning(model_output, None)
    message_vllm = {
        "content": content,
        "reasoning": reasoning,
    }

    wrapper = VLLMParserWrapper(parser)
    message = {"content": model_output}
    wrapper.parse(message)  # modifies message in place

    assert compare_dicts(message, message_vllm, skip_keys=["id"])


class _ReasoningParserReturningNone:
    """Follows vLLM's BaseThinkingReasoningParser.extract_reasoning: a missing part is None."""

    def extract_reasoning(self, model_output, request):
        if "</think>" not in model_output:
            return model_output, None
        reasoning, _, content = model_output.partition("</think>")
        return reasoning, content or None


@pytest.mark.parametrize(
    "model_output, expected",
    [
        ("thinking</think>answer", {"content": "answer", "reasoning": "thinking"}),
        ("thinking</think>", {"content": None, "reasoning": "thinking"}),
        ("still thinking", {"content": None, "reasoning": "still thinking"}),
    ],
)
def test_wrapper_keeps_none_from_extract_reasoning(model_output, expected):
    wrapper = VLLMParserWrapper(_ReasoningParserReturningNone())
    message = {"content": model_output}
    wrapper.parse(message)
    assert message == expected


class _ReasoningParserReturning:
    def __init__(self, result):
        self.result = result

    def extract_reasoning(self, model_output, request):
        return self.result


@pytest.mark.parametrize(
    "result, expected",
    [
        ((None, "answer"), {"content": "answer", "reasoning": None}),
        ((None, None), {"content": None, "reasoning": None}),
        (("", ""), {"content": "", "reasoning": ""}),
    ],
)
def test_wrapper_keeps_reasoning_parts_as_returned(result, expected):
    message = {"content": "text"}
    VLLMParserWrapper(_ReasoningParserReturning(result)).parse(message)
    assert message == expected


@pytest.mark.parametrize("result", [("a", "b", "c"), ("a",), ["a", "b"], None])
def test_wrapper_rejects_result_that_is_not_a_pair(result):
    with pytest.raises(RuntimeError, match="not a tuple of length 2"):
        VLLMParserWrapper(_ReasoningParserReturning(result)).parse({"content": "text"})


def test_wrapper_rejects_parts_that_are_not_strings():
    with pytest.raises(RuntimeError):
        VLLMParserWrapper(_ReasoningParserReturning(("a", 2))).parse({"content": "text"})
