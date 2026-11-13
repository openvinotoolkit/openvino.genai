# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the text size helpers in ``llm_bench_utils.gen_output_data``."""

import pytest

from llm_bench_utils.gen_output_data import (
    count_words,
    embed_iterate_data,
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
    assert text_output_repr("...") == "text:0w"


def test_text_output_repr_empty_output():
    assert text_output_repr("") == "empty"
    assert text_output_repr("  \n\t") == "empty"
    assert text_output_repr(None) == "empty"
    assert text_output_repr("", 3) == "empty"


def test_gen_iterate_data_stores_text_tokens():
    assert gen_iterate_data(in_text_tokens=7)["input_text_tokens"] == 7
    assert gen_iterate_data(in_text_tokens=None)["input_text_tokens"] == ""
    assert gen_iterate_data()["input_text_tokens"] == ""


def test_text_output_repr_appends_token_count():
    assert text_output_repr("Hello , world .", 4) == "text:2w/4t"
    assert text_output_repr("...", 1) == "text:0w/1t"
    assert text_output_repr("Hello world", 0) == "text:2w/0t"


def test_text_output_repr_drops_unknown_token_count():
    assert text_output_repr("Hello world", None) == "text:2w"


def test_embed_iterate_data_output_repr():
    assert embed_iterate_data()["output_repr"] == ""  # embeddings
    assert embed_iterate_data(output_repr="docs:2")["output_repr"] == "docs:2"  # rerankers


def test_embed_iterate_data_input_text_tokens():
    assert embed_iterate_data(in_text_tokens=5)["input_text_tokens"] == 5  # rerankers
    assert embed_iterate_data(in_text_tokens=None)["input_text_tokens"] == ""
    assert embed_iterate_data()["input_text_tokens"] == ""  # embeddings
