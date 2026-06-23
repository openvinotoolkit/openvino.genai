# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import types

import pytest

from whowhatbench import wwb


class _Tokenizer:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template


class _Processor:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template


def _make_args():
    return types.SimpleNamespace(base_model="dummy", target_model=None)


def test_processor_chat_template_returned_when_available(monkeypatch):
    monkeypatch.setattr(wwb, "load_processor", lambda args: (_Processor("TEMPLATE"), None))
    assert wwb.get_processor_chat_template(_make_args()) == "TEMPLATE"


def test_processor_chat_template_none_when_processor_has_no_template(monkeypatch):
    monkeypatch.setattr(wwb, "load_processor", lambda args: (_Processor(None), None))
    assert wwb.get_processor_chat_template(_make_args()) is None


def test_processor_chat_template_none_when_processor_missing(monkeypatch):
    monkeypatch.setattr(wwb, "load_processor", lambda args: (None, None))
    assert wwb.get_processor_chat_template(_make_args()) is None


def test_processor_chat_template_none_when_load_fails(monkeypatch):
    def _raise(args):
        raise RuntimeError("processor cannot be loaded for a non-multimodal model")

    monkeypatch.setattr(wwb, "load_processor", _raise)
    # A processor-loading failure must be swallowed and reported as "no template".
    assert wwb.get_processor_chat_template(_make_args()) is None
