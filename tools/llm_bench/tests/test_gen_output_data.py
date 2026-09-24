# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the text size helpers in ``llm_bench_utils.gen_output_data``."""

import types

import numpy as np
import pytest

from llm_bench_utils.gen_output_data import (
    count_text_tokens,
    count_words,
    gen_iterate_data,
    text_output_repr,
)


# --------------------------------------------------------------------------- #
# count_words                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, world.", 2),
        ("Hello , world .", 2),  # standalone punctuation is not a word
        ("- item one", 2),
        ("don't", 1),
        ("don’t", 1),  # typographic apostrophe
        ("state-of-the-art", 1),
        ("snake_case", 2),  # underscore separates, like punctuation
        ("  one\ttwo\n\nthree  ", 3),
        ("x = 3.14", 3),
        ("这是一个测试", 6),  # one word per CJK character
        ("カタカナ", 4),
        ("GPU加速", 3),  # a Latin run and each CJK character are separate
        ("Zażółć gęślą jaźń", 3),  # non-ASCII letters are word characters
        ("...", 0),
        ("", 0),
        (None, 0),
    ],
)
def test_count_words(text, expected):
    assert count_words(text) == expected


def test_text_output_repr_uses_count_words():
    assert text_output_repr("Hello , world .") == "text:2w"
    assert text_output_repr("...") == ""


# --------------------------------------------------------------------------- #
# count_text_tokens                                                            #
# --------------------------------------------------------------------------- #


class FakeHFTokenizer:
    """Mimics a HF tokenizer: ``encode`` returns a plain list of ids."""

    def __init__(self):
        self.kwargs = None

    def encode(self, text, **kwargs):
        self.kwargs = kwargs
        return list(range(len(text.split())))


class FakeGenAITokenizer:
    """Mimics ``openvino_genai.Tokenizer``: ``encode`` returns TokenizedInputs."""

    def encode(self, text, **kwargs):
        return types.SimpleNamespace(input_ids=np.zeros((1, 4 * len(text.split()))))


def test_count_text_tokens_hf_tokenizer_skips_special_tokens():
    tok = FakeHFTokenizer()
    assert count_text_tokens(tok, "a b c") == 3
    assert tok.kwargs == {"add_special_tokens": False}


def test_count_text_tokens_genai_tokenizer():
    assert count_text_tokens(FakeGenAITokenizer(), "a b") == 8


def test_count_text_tokens_unwraps_a_processor():
    processor = types.SimpleNamespace(tokenizer=FakeHFTokenizer())
    assert count_text_tokens(processor, "a b") == 2


def test_count_text_tokens_is_none_when_unknown():
    class Broken:
        def encode(self, text, **kwargs):
            raise RuntimeError("no encode for you")

    assert count_text_tokens(Broken(), "a b") is None
    assert count_text_tokens(None, "a b") is None
    assert count_text_tokens(FakeHFTokenizer(), "") is None


def test_gen_iterate_data_stores_text_tokens():
    assert gen_iterate_data(in_text_tokens=7)["input_text_tokens"] == 7
    assert gen_iterate_data(in_text_tokens=None)["input_text_tokens"] == ""
    assert gen_iterate_data()["input_text_tokens"] == ""


def test_text_output_repr_appends_token_count():
    assert text_output_repr("Hello , world .", FakeHFTokenizer()) == "text:2w/4t"
    assert text_output_repr("Hello world", FakeGenAITokenizer()) == "text:2w/8t"
    assert text_output_repr("...", FakeHFTokenizer()) == ""


def test_text_output_repr_drops_unknown_token_count():
    class Broken:
        def encode(self, text, **kwargs):
            raise RuntimeError("no encode for you")

    assert text_output_repr("Hello world", Broken()) == "text:2w"
    assert text_output_repr("Hello world", None) == "text:2w"
