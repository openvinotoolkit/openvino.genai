# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest


class FakeDecodedResults:
    """Mimics openvino_genai's DecodedResults, which exposes a .texts list."""

    def __init__(self, texts):
        self.texts = texts


class FakeModel:
    def __init__(self, answer):
        self._answer = answer

    def generate(self, *args, **kwargs):
        return self._answer


def _call_genai_gen_text(answer):
    from whowhatbench.wwb import genai_gen_text

    model = FakeModel(answer)
    return genai_gen_text(
        model,
        tokenizer=None,
        question="question",
        max_new_tokens=16,
        skip_question=False,
    )


def test_genai_gen_text_unwraps_single_text_result():
    """A DecodedResults-like object with exactly one text is unwrapped to a plain string."""
    answer = FakeDecodedResults(["only answer"])
    assert _call_genai_gen_text(answer) == "only answer"


def test_genai_gen_text_keeps_multi_text_result_as_is():
    """A DecodedResults-like object with more than one text is normalized to the first sequence."""
    answer = FakeDecodedResults(["first", "second"])
    result = _call_genai_gen_text(answer)
    assert result == "first"
    assert isinstance(result, str)


def test_genai_gen_text_unwraps_singleton_string_list():
    """A plain list containing exactly one string is unwrapped to that string."""
    answer = ["only answer"]
    assert _call_genai_gen_text(answer) == "only answer"


def test_genai_gen_text_keeps_multi_item_string_list_as_is():
    """A plain list containing more than one string is normalized to the first item."""
    answer = ["first", "second"]
    result = _call_genai_gen_text(answer)
    assert result == "first"
    assert isinstance(result, str)


def test_genai_gen_text_returns_plain_string_unchanged():
    """A plain string answer (typical LLMPipeline.generate() result) passes through untouched."""
    answer = "plain string answer"
    assert _call_genai_gen_text(answer) == "plain string answer"


def test_genai_gen_text_handles_none_texts_attribute():
    """An object whose .texts attribute is None can't be normalized to a str and must raise TypeError."""

    class AnswerWithNoneTexts:
        texts = None

    answer = AnswerWithNoneTexts()
    with pytest.raises(TypeError):
        _call_genai_gen_text(answer)


def test_genai_gen_text_handles_string_texts_attribute():
    """An object whose .texts attribute is itself a string (not a list/tuple) must raise TypeError,
    not be indexed char-by-char."""

    class AnswerWithStringTexts:
        texts = "abc"

    answer = AnswerWithStringTexts()
    with pytest.raises(TypeError):
        _call_genai_gen_text(answer)


def test_genai_gen_text_unwraps_texts_tuple():
    """A .texts tuple with exactly one item is unwrapped, matching list handling."""

    class AnswerWithTupleTexts:
        texts = ("only answer",)

    answer = AnswerWithTupleTexts()
    assert _call_genai_gen_text(answer) == "only answer"


def test_genai_gen_text_raises_on_unnormalizable_answer():
    """An answer with no .texts and that isn't a str/list must raise TypeError, never be returned as-is."""
    answer = object()
    with pytest.raises(TypeError):
        _call_genai_gen_text(answer)


def test_genai_gen_text_raises_on_empty_string_list():
    """An empty list can't be normalized to a single str and must raise TypeError."""
    with pytest.raises(TypeError):
        _call_genai_gen_text([])
