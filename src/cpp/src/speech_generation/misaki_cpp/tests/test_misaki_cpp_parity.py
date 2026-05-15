# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path

import pytest


logger = logging.getLogger(__name__)


misaki_cpp_py = pytest.importorskip(
    "misaki_cpp_py", reason="misaki_cpp_py module is not available; build with MISAKI_CPP_BUILD_PYTHON_BINDINGS=ON"
)

try:
    import misaki
    from misaki import en as misaki_en
    from misaki import espeak as misaki_espeak

    HAS_MISAKI_PYTHON = True
except ImportError:
    HAS_MISAKI_PYTHON = False


def _configure_espeak_from_venv() -> None:
    """
    Point the C++ misaki espeak fallback at the DLL and data directory that
    come bundled with the ``espeakng_loader`` Python package (installed as a
    transitive dependency of ``misaki``).  This is necessary on CI machines
    where espeak-ng is not installed in a system location.

    Sets two environment variables **only if they are not already set**:
    * ``MISAKI_ESPEAK_LIBRARY`` – absolute path to the espeak-ng shared library
      (read by ``kokoro_tts_model.cpp`` via ``std::getenv``).
    * ``ESPEAK_DATA_PATH`` – directory containing ``espeak-ng-data/``
      (read by the espeak-ng runtime itself during ``espeak_Initialize``).
    """
    try:
        import espeakng_loader  # type: ignore[import]
    except ImportError:
        logger.debug("espeakng_loader not available; skipping espeak env setup")
        return

    lib_path = espeakng_loader.get_library_path()
    data_path = espeakng_loader.get_data_path()

    if "MISAKI_ESPEAK_LIBRARY" not in os.environ:
        os.environ["MISAKI_ESPEAK_LIBRARY"] = lib_path
        logger.debug("Set MISAKI_ESPEAK_LIBRARY=%s", lib_path)

    if "ESPEAK_DATA_PATH" not in os.environ:
        os.environ["ESPEAK_DATA_PATH"] = data_path
        logger.debug("Set ESPEAK_DATA_PATH=%s", data_path)


# Configure espeak at import time so the env vars are in place before
# any Engine instances are created.
_configure_espeak_from_venv()


# Multilingual test cases with fallback and non-English coverage
MULTILINGUAL_PROMPT_CASES = [
    ("Hello this is a short speech generation test.", "en-us"),
    ("Today we analyse colour and flavour.", "en-gb"),
    ("Hola esto es una prueba de voz sintetica.", "es"),
    ("Bonjour ceci est un test de synthese vocale.", "fr-fr"),
    ("नमस्ते यह वाक संश्लेषण परीक्षण है।", "hi"),
    ("Ciao questo e un test di sintesi vocale.", "it"),
    ("Ola isto e um teste de sintese de voz.", "pt-br"),
]
MULTILINGUAL_CASE_IDS = [f"multilingual-{language}" for _, language in MULTILINGUAL_PROMPT_CASES]


def _misaki_lexicon_data_root() -> Path:
    """Return lexicon root, preferring installed misaki package data when available."""
    if HAS_MISAKI_PYTHON:
        misaki_module_path = Path(misaki.__file__).resolve().parent
        installed_data_root = misaki_module_path / "data"
        if installed_data_root.exists():
            return installed_data_root

    local_data_root = Path(__file__).resolve().parents[1] / "data"
    if local_data_root.exists():
        return local_data_root

    raise FileNotFoundError("Could not locate misaki lexicon data in installed package or local source tree")


def _resolve_cpp_engine_lang_and_variant(language_variant: str) -> tuple[str, str]:
    if language_variant in ("en-us", "en-gb"):
        return "en", language_variant
    return "espeak", language_variant


def _build_cpp_engine(language_variant: str):
    lang, variant = _resolve_cpp_engine_lang_and_variant(language_variant)
    engine = misaki_cpp_py.Engine(lang, variant)
    engine.set_lexicon_data_root(str(_misaki_lexicon_data_root()))
    return engine


def _build_python_g2p(language_variant: str):
    if language_variant in ("en-us", "en-gb"):
        british = language_variant == "en-gb"
        fallback = misaki_espeak.EspeakFallback(british=british)
        return misaki_en.G2P(trf=False, british=british, fallback=fallback, unk="❓"), True

    return misaki_espeak.EspeakG2P(language_variant), False


def _python_phonemize(language_variant: str, text: str) -> str:
    g2p, _ = _build_python_g2p(language_variant)
    phonemes, _tokens_or_none = g2p(text)
    return phonemes


def _python_phonemize_with_tokens(language_variant: str, text: str) -> tuple[str, list]:
    g2p, supports_tokens = _build_python_g2p(language_variant)
    phonemes, tokens_or_none = g2p(text)
    if not supports_tokens:
        raise RuntimeError(f"Python misaki backend for variant '{language_variant}' does not provide token output")
    return phonemes, tokens_or_none


def _normalize_phonemes_for_parity(phonemes: str) -> str:
    """Normalize canonical-equivalent affricate spellings used by different backends."""
    return phonemes.replace("tʃ", "ʧ").replace("dʒ", "ʤ")


def test_misaki_cpp_engine_phonemize_returns_text() -> None:
    engine = _build_cpp_engine("en-us")

    phonemes = engine.phonemize("hello world")

    assert isinstance(phonemes, str)
    assert phonemes.strip(), "Expected non-empty phoneme output"


def test_misaki_cpp_engine_phonemize_with_tokens_shape() -> None:
    engine = _build_cpp_engine("en-us")

    result = engine.phonemize_with_tokens("hello world")

    assert isinstance(result, dict)
    assert "phonemes" in result
    assert isinstance(result["phonemes"], str)
    assert "tokens" in result
    assert isinstance(result["tokens"], list)
    assert result["tokens"], "Expected at least one token in output"

    token = result["tokens"][0]
    assert isinstance(token, dict)
    for field_name in ["text", "tag", "whitespace", "phonemes", "start_ts", "end_ts", "_"]:
        assert field_name in token, f"Missing token field: {field_name}"


def test_misaki_cpp_engine_invalid_lexicon_root_raises() -> None:
    engine = misaki_cpp_py.Engine("en", "en-us")
    engine.set_lexicon_data_root(str(Path("/definitely/nonexistent/misaki_lexicon_root")))

    with pytest.raises(RuntimeError, match="Could not locate English lexicon data files"):
        engine.phonemize("hello")

    # Clear the global override to avoid leaking state to unrelated tests.
    engine.clear_lexicon_data_root()


@pytest.mark.skipif(not HAS_MISAKI_PYTHON, reason="misaki Python module not available")
def test_misaki_cpp_vs_python_simple_sentence() -> None:
    """Compare misaki C++ bindings with misaki Python module on simple sentence."""
    test_text = "Hello there."

    cpp_engine = _build_cpp_engine("en-us")
    cpp_phonemes = cpp_engine.phonemize(test_text)

    py_phonemes = _python_phonemize("en-us", test_text)

    assert cpp_phonemes.strip(), "C++ engine produced empty phonemes"
    assert py_phonemes.strip(), "Python engine produced empty phonemes"

    cpp_norm = _normalize_phonemes_for_parity(cpp_phonemes)
    py_norm = _normalize_phonemes_for_parity(py_phonemes)
    assert cpp_norm == py_norm, (
        "Phoneme mismatch after normalization:\n"
        f"C++ raw: {cpp_phonemes}\n"
        f"Python raw: {py_phonemes}\n"
        f"C++ norm: {cpp_norm}\n"
        f"Python norm: {py_norm}"
    )


@pytest.mark.skipif(not HAS_MISAKI_PYTHON, reason="misaki Python module not available")
def test_misaki_cpp_vs_python_with_tokens() -> None:
    """Compare misaki C++ bindings with misaki Python module on phonemize_with_tokens."""
    test_text = "Good morning."

    cpp_engine = _build_cpp_engine("en-us")
    cpp_result = cpp_engine.phonemize_with_tokens(test_text)

    py_phonemes, py_tokens = _python_phonemize_with_tokens("en-us", test_text)

    cpp_norm = _normalize_phonemes_for_parity(cpp_result["phonemes"])
    py_norm = _normalize_phonemes_for_parity(py_phonemes)
    assert cpp_norm == py_norm, (
        "Phoneme mismatch after normalization:\n"
        f"C++ raw: {cpp_result['phonemes']}\n"
        f"Python raw: {py_phonemes}\n"
        f"C++ norm: {cpp_norm}\n"
        f"Python norm: {py_norm}"
    )

    cpp_token_phonemes = [(token.get("phonemes") or "") for token in cpp_result["tokens"]]
    py_token_phonemes = [((token.phonemes or "") if token is not None else "") for token in py_tokens]
    cpp_token_norm = [_normalize_phonemes_for_parity(p) for p in cpp_token_phonemes]
    py_token_norm = [_normalize_phonemes_for_parity(p) for p in py_token_phonemes]
    assert cpp_token_norm == py_token_norm, (
        "Token phoneme stream mismatch after normalization:\n"
        f"C++ raw: {cpp_token_phonemes}\n"
        f"Python raw: {py_token_phonemes}\n"
        f"C++ norm: {cpp_token_norm}\n"
        f"Python norm: {py_token_norm}"
    )


@pytest.mark.skipif(not HAS_MISAKI_PYTHON, reason="misaki Python module not available")
@pytest.mark.parametrize("text,language", MULTILINGUAL_PROMPT_CASES, ids=MULTILINGUAL_CASE_IDS)
def test_misaki_cpp_vs_python_multilingual(text: str, language: str) -> None:
    """Compare misaki C++ and Python engines on multilingual prompts with fallback coverage."""
    cpp_engine = _build_cpp_engine(language)
    cpp_phonemes = cpp_engine.phonemize(text)

    py_phonemes = _python_phonemize(language, text)

    assert cpp_phonemes.strip(), f"C++ engine produced empty phonemes for {language}: {text}"
    assert py_phonemes.strip(), f"Python engine produced empty phonemes for {language}: {text}"

    cpp_norm = _normalize_phonemes_for_parity(cpp_phonemes)
    py_norm = _normalize_phonemes_for_parity(py_phonemes)
    assert cpp_norm == py_norm, (
        f"Phoneme mismatch for {language} after normalization:\n"
        f"Text: {text}\n"
        f"C++ raw: {cpp_phonemes}\n"
        f"Python raw: {py_phonemes}\n"
        f"C++ norm: {cpp_norm}\n"
        f"Python norm: {py_norm}"
    )


@pytest.mark.skipif(not HAS_MISAKI_PYTHON, reason="misaki Python module not available")
@pytest.mark.parametrize(
    "text,language",
    [
        ("Hello this is a short speech generation test.", "en-us"),
        ("Today we analyse colour and flavour.", "en-gb"),
    ],
    ids=["english-with-tokens-en-us", "english-with-tokens-en-gb"],
)
def test_misaki_cpp_vs_python_english_with_tokens(text: str, language: str) -> None:
    """Compare misaki C++ and Python with token output on English variants."""
    cpp_engine = _build_cpp_engine(language)
    cpp_result = cpp_engine.phonemize_with_tokens(text)

    py_phonemes, py_tokens = _python_phonemize_with_tokens(language, text)

    cpp_norm = _normalize_phonemes_for_parity(cpp_result["phonemes"])
    py_norm = _normalize_phonemes_for_parity(py_phonemes)
    assert cpp_norm == py_norm, (
        f"Phoneme mismatch for {language} after normalization:\n"
        f"Text: {text}\n"
        f"C++ raw: {cpp_result['phonemes']}\n"
        f"Python raw: {py_phonemes}\n"
        f"C++ norm: {cpp_norm}\n"
        f"Python norm: {py_norm}"
    )

    cpp_token_phonemes = [(token.get("phonemes") or "") for token in cpp_result["tokens"]]
    py_token_phonemes = [((token.phonemes or "") if token is not None else "") for token in py_tokens]
    cpp_token_norm = [_normalize_phonemes_for_parity(p) for p in cpp_token_phonemes]
    py_token_norm = [_normalize_phonemes_for_parity(p) for p in py_token_phonemes]
    assert cpp_token_norm == py_token_norm, (
        f"Token phoneme stream mismatch for {language} after normalization:\n"
        f"C++ raw: {cpp_token_phonemes}\n"
        f"Python raw: {py_token_phonemes}\n"
        f"C++ norm: {cpp_token_norm}\n"
        f"Python norm: {py_token_norm}"
    )
