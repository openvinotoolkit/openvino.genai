# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import itertools

import pytest

import openvino_genai


def _get_kokoro_model_dir() -> pathlib.Path:
    model_dir = os.getenv("KOKORO_OV_MODEL_DIR")
    if not model_dir:
        pytest.skip("Set KOKORO_OV_MODEL_DIR to run Kokoro phonemize parity tests")
    path = pathlib.Path(model_dir)
    if not path.exists():
        pytest.skip(f"KOKORO_OV_MODEL_DIR does not exist: {path}")
    return path


def _native_en_chunks(text: str, lang_code: str):
    kokoro = pytest.importorskip("kokoro")

    native_pipe = kokoro.KPipeline(lang_code=lang_code, model=False)
    _, tokens = native_pipe.g2p(text)

    chunks = []
    for _, ps, _ in native_pipe.en_tokenize(tokens):
        if ps:
            chunks.append(ps)
    return chunks


LONG_TEXT_CASES = [
    "At 07:45 a.m. on Feb 26, 2026, I stepped off the NovaLiner-3 at Aeroluna Station and met Dr. Vellorin Quade from the Lumenfield R&D Lab.",
    "In the valley of Luminara, a brindleflit named Marilo followed the silverwinding road to Coralhaven while glimmerpines hummed and whisperwillows swayed.",
    "The OpenVINO GenAI repository is licensed under Apache License Version 2.0, and this sentence intentionally includes punctuation, numerals, and long-word OOD tokens like twinklepack and windleberries.",
]

MAX_PHONEME_LENGTH = 510


def _get_speech_token_type():
    token_type = getattr(openvino_genai, "SpeechToken", None)
    if token_type is None:
        pytest.skip("SpeechToken is not exported by openvino_genai in this build")
    return token_type


@pytest.mark.speech_generation
@pytest.mark.parametrize("language,lang_code", [("en-us", "a"), ("en-gb", "b")])
@pytest.mark.parametrize("text", LONG_TEXT_CASES)
def test_kokoro_phonemize_parity_long_text(language: str, lang_code: str, text: str):
    model_dir = _get_kokoro_model_dir()

    ov_pipe = openvino_genai.Text2SpeechPipeline(model_dir, "CPU")

    ov_chunks = ov_pipe.phonemize(
        text,
        speech_model_type="kokoro",
        language=language,
        max_phoneme_length=MAX_PHONEME_LENGTH,
    )

    native_chunks = _native_en_chunks(text, lang_code)
    assert len(ov_chunks) == len(native_chunks), (
        f"Chunk count mismatch for language={language}: OV={len(ov_chunks)} vs native={len(native_chunks)}"
    )

    ov_total = sum(len(chunk) for chunk in ov_chunks)
    native_total = sum(len(chunk) for chunk in native_chunks)
    assert native_total > 0
    coverage_ratio = ov_total / native_total
    assert 0.30 <= coverage_ratio <= 2.50, (
        f"Total phoneme coverage ratio out of bounds for language={language}: "
        f"OV/native={coverage_ratio:.3f} (OV={ov_total}, native={native_total})"
    )


@pytest.mark.speech_generation
@pytest.mark.parametrize("language,lang_code", [("en-us", "a"), ("en-gb", "b")])
@pytest.mark.parametrize("text", LONG_TEXT_CASES)
def test_kokoro_phonemize_chunk_invariants_long_text(language: str, lang_code: str, text: str):
    model_dir = _get_kokoro_model_dir()

    ov_pipe = openvino_genai.Text2SpeechPipeline(model_dir, "CPU")

    ov_chunks = ov_pipe.phonemize(
        text,
        speech_model_type="kokoro",
        language=language,
        max_phoneme_length=MAX_PHONEME_LENGTH,
    )
    native_chunks = _native_en_chunks(text, lang_code)

    assert len(ov_chunks) == len(native_chunks), (
        f"Chunk count mismatch for language={language}: OV={len(ov_chunks)} vs native={len(native_chunks)}"
    )

    for idx, (ov_chunk, native_chunk) in enumerate(itertools.zip_longest(ov_chunks, native_chunks, fillvalue="")):
        ov_len = len(ov_chunk)
        native_len = len(native_chunk)
        assert ov_len <= MAX_PHONEME_LENGTH, (
            f"OV chunk[{idx}] length exceeds max_phoneme_length={MAX_PHONEME_LENGTH}: {ov_len}"
        )
        assert native_len <= MAX_PHONEME_LENGTH, (
            f"Native chunk[{idx}] length exceeds max_phoneme_length={MAX_PHONEME_LENGTH}: {native_len}"
        )
        if native_len == 0:
            assert ov_len == 0
            continue

        per_chunk_ratio = ov_len / native_len
        assert 0.30 <= per_chunk_ratio <= 2.50, (
            f"Chunk[{idx}] length ratio out of bounds for language={language}: "
            f"OV/native={per_chunk_ratio:.3f} (OV={ov_len}, native={native_len})"
        )


@pytest.mark.speech_generation
def test_kokoro_phonemize_batch_api_shape():
    model_dir = _get_kokoro_model_dir()

    ov_pipe = openvino_genai.Text2SpeechPipeline(model_dir, "CPU")
    texts = LONG_TEXT_CASES[:2]

    chunk_lists = ov_pipe.phonemize(
        texts,
        speech_model_type="kokoro",
        language="en-us",
        max_phoneme_length=MAX_PHONEME_LENGTH,
    )

    assert isinstance(chunk_lists, list)
    assert len(chunk_lists) == len(texts)
    assert all(isinstance(chunks, list) for chunks in chunk_lists)
    assert all(len(chunks) > 0 for chunks in chunk_lists)


@pytest.mark.speech_generation
def test_kokoro_generate_from_tokens_single_sequence():
    model_dir = _get_kokoro_model_dir()

    ov_pipe = openvino_genai.Text2SpeechPipeline(model_dir, "CPU")
    token_type = _get_speech_token_type()
    if token_type is None:
        pytest.skip("SpeechToken is not available in this build")

    tokens = [
        token_type("həlˈoʊ", True, "Hello"),
        token_type("wˈɝld", False, "world"),
    ]

    result = ov_pipe.generate_from_tokens(tokens, None, voice="af_heart", language="en-us")

    assert len(result.speeches) == 1
    assert result.output_sample_rate > 0
    assert result.speeches[0].get_size() > 0


@pytest.mark.speech_generation
def test_kokoro_generate_from_tokens_batch_shape():
    model_dir = _get_kokoro_model_dir()

    ov_pipe = openvino_genai.Text2SpeechPipeline(model_dir, "CPU")
    token_type = _get_speech_token_type()
    if token_type is None:
        pytest.skip("SpeechToken is not available in this build")

    token_batches = [
        [
            token_type("həlˈoʊ", True, "Hello"),
            token_type("wˈɝld", False, "world"),
        ],
        [
            token_type("ðæŋks", True, "Thanks"),
            token_type("aʊt", False, "out"),
        ],
    ]

    result = ov_pipe.generate_from_tokens(token_batches, None, voice="af_heart", language="en-us")

    assert len(result.speeches) == len(token_batches)
    assert result.output_sample_rate > 0
    assert all(speech.get_size() > 0 for speech in result.speeches)
