# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


class FakeTokenizer:
    def __init__(self, chat_template):
        self.chat_template = chat_template
        self.set_calls = []

    def set_chat_template(self, chat_template):
        self.set_calls.append(chat_template)
        self.chat_template = chat_template


class FakePipeline:
    def __init__(self, chat_template):
        self.tokenizer = FakeTokenizer(chat_template)

    def get_tokenizer(self):
        return self.tokenizer


def test_regex_joins_adjacent_multiline_string_literals():
    from whowhatbench.model_loaders import _MINJA_MULTILINE_STRING_CONCAT_RE

    broken_template = (
        "raise_exception(\n"
        '    "chat_template: tool_calls[].function.arguments must be a "\n'
        '    "JSON object (mapping), not a string. Deserialize arguments "\n'
        '    "before passing to the template."\n'
        ")"
    )

    patched = _MINJA_MULTILINE_STRING_CONCAT_RE.sub("", broken_template)

    assert patched == (
        "raise_exception(\n"
        '    "chat_template: tool_calls[].function.arguments must be a '
        "JSON object (mapping), not a string. Deserialize arguments "
        'before passing to the template."\n'
        ")"
    )


def test_patch_updates_tokenizer_when_template_is_minja_incompatible():
    from whowhatbench.model_loaders import _patch_minja_incompatible_chat_template

    broken_template = '{{ raise_exception("first "\n"second") }}'
    pipeline = FakePipeline(broken_template)

    _patch_minja_incompatible_chat_template(pipeline)

    assert pipeline.tokenizer.chat_template == '{{ raise_exception("first second") }}'
    assert pipeline.tokenizer.set_calls == ['{{ raise_exception("first second") }}']


def test_patch_is_noop_for_already_compatible_template():
    from whowhatbench.model_loaders import _patch_minja_incompatible_chat_template

    normal_template = '{{ raise_exception("a single line message") }}'
    pipeline = FakePipeline(normal_template)

    _patch_minja_incompatible_chat_template(pipeline)

    assert pipeline.tokenizer.chat_template == normal_template
    assert pipeline.tokenizer.set_calls == []


def test_patch_is_noop_for_empty_chat_template():
    from whowhatbench.model_loaders import _patch_minja_incompatible_chat_template

    pipeline = FakePipeline("")

    _patch_minja_incompatible_chat_template(pipeline)

    assert pipeline.tokenizer.chat_template == ""
    assert pipeline.tokenizer.set_calls == []


def test_patch_is_noop_for_none_chat_template():
    from whowhatbench.model_loaders import _patch_minja_incompatible_chat_template

    pipeline = FakePipeline(None)

    _patch_minja_incompatible_chat_template(pipeline)

    assert pipeline.tokenizer.chat_template is None
    assert pipeline.tokenizer.set_calls == []


def test_patch_is_noop_for_non_string_chat_template():
    from whowhatbench.model_loaders import _patch_minja_incompatible_chat_template

    # Some tokenizers may expose chat_template as a non-string (e.g. dict of named templates).
    non_string_template = {"default": '{{ raise_exception("first "\n"second") }}'}
    pipeline = FakePipeline(non_string_template)

    # Must not raise.
    _patch_minja_incompatible_chat_template(pipeline)

    assert pipeline.tokenizer.chat_template == non_string_template
    assert pipeline.tokenizer.set_calls == []


def test_patch_swallows_get_tokenizer_errors():
    from whowhatbench.model_loaders import _patch_minja_incompatible_chat_template

    class PipelineWithoutTokenizer:
        def get_tokenizer(self):
            raise RuntimeError("get_tokenizer() not supported for this pipeline")

    # Must not raise.
    _patch_minja_incompatible_chat_template(PipelineWithoutTokenizer())


def test_patch_swallows_set_chat_template_errors():
    from whowhatbench.model_loaders import _patch_minja_incompatible_chat_template

    class TokenizerRaisingOnSet:
        chat_template = '{{ raise_exception("first "\n"second") }}'

        def set_chat_template(self, chat_template):
            raise RuntimeError("cannot set chat template")

    class PipelineWithTokenizer:
        def get_tokenizer(self):
            return TokenizerRaisingOnSet()

    # Must not raise even though set_chat_template() fails.
    _patch_minja_incompatible_chat_template(PipelineWithTokenizer())
