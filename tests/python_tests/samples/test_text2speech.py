# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, SAMPLES_JS_DIR
from test_utils import run_sample
from utils.constants import get_ov_cache_converted_models_dir
from utils.kokoro_test_assets import prepare_tiny_g2p_model_path
from utils.kokoro_test_assets import prepare_tiny_g2p_ov_path
from utils.kokoro_test_assets import prepare_tiny_kokoro_model_path
from utils.kokoro_test_assets import prepare_tiny_kokoro_ov_path

rng = np.random.default_rng(34231)


@pytest.fixture(scope="module")
def tiny_kokoro_ov_path() -> Path:
    converted_models_dir = get_ov_cache_converted_models_dir()
    tiny_kokoro_model_path = prepare_tiny_kokoro_model_path(converted_models_dir)
    return prepare_tiny_kokoro_ov_path(converted_models_dir, tiny_kokoro_model_path)


@pytest.fixture(scope="module")
def tiny_g2p_ov_path() -> Path:
    converted_models_dir = get_ov_cache_converted_models_dir()
    tiny_g2p_model_path = prepare_tiny_g2p_model_path(converted_models_dir)
    return prepare_tiny_g2p_ov_path(converted_models_dir, tiny_g2p_model_path)


@pytest.fixture(scope="module")
def tiny_kokoro_speaker_embedding_file_path(tiny_kokoro_ov_path: Path) -> str:
    voice_bin_path = tiny_kokoro_ov_path / "voices" / "tiny_voice.bin"
    if not voice_bin_path.exists():
        raise FileNotFoundError(f"Missing tiny Kokoro speaker embedding file at {voice_bin_path}")
    return str(voice_bin_path)


class TestTextToSpeechSample:
    def setup_class(self):
        # Create temporary binary file containing speaker embedding
        self.temp_speaker_embedding_file = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
        # Generate 512 random float32 values
        data = rng.random(512, dtype=np.float32)
        # Write to file
        data.tofile(self.temp_speaker_embedding_file)
        self.temp_speaker_embedding_file.close()

    def teardown_class(self):
        # Remove temporary file
        if os.path.exists(self.temp_speaker_embedding_file.name):
            os.remove(self.temp_speaker_embedding_file.name)

    @pytest.mark.speech_generation
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["tiny-random-SpeechT5ForTextToSpeech"], indirect=True)
    @pytest.mark.parametrize("input_prompt", ["Hello everyone"])
    def test_sample_text_to_speech(self, convert_model, input_prompt):
        # Example: text2speech spt5_model_dir "Hello everyone" --speaker_embedding_file_path xvector.bin
        # Run C++ sample
        cpp_sample = SAMPLES_CPP_DIR / 'text2speech'
        cpp_command = [cpp_sample, convert_model, input_prompt, self.temp_speaker_embedding_file.name]
        cpp_result = run_sample(cpp_command)

        # Run Python sample
        py_script = SAMPLES_PY_DIR / "speech_generation/text2speech.py"
        py_command = [sys.executable, py_script, convert_model, input_prompt,
                      "--speaker_embedding_file_path", self.temp_speaker_embedding_file.name]
        py_result = run_sample(py_command)

        # Run JS sample
        js_script = SAMPLES_JS_DIR / "speech_generation/text2speech.js"
        js_command = [
            "node",
            js_script,
            convert_model,
            input_prompt,
            "--speaker_embedding",
            self.temp_speaker_embedding_file.name,
        ]
        js_result = run_sample(js_command)

        assert "Text successfully converted to audio file" in cpp_result.stdout, (
            "C++ sample text2speech must be successfully completed"
        )
        assert "Text successfully converted to audio file" in py_result.stdout, (
            "Python sample text2speech must be successfully completed"
        )
        assert "Text successfully converted to audio file" in js_result.stdout, (
            "JS sample text2speech must be successfully completed"
        )

    @pytest.mark.speech_generation
    @pytest.mark.samples
    @pytest.mark.parametrize("input_prompt", ["Hello, and welcome to speech generation using OpenVINO GenAI."])
    def test_sample_text_to_speech_kokoro(
        self,
        tiny_kokoro_ov_path: Path,
        tiny_kokoro_speaker_embedding_file_path: str,
        input_prompt: str,
    ):
        # Run C++ sample with Kokoro model + language + explicit speaker embedding.
        cpp_sample = SAMPLES_CPP_DIR / "text2speech"
        cpp_command = [
            cpp_sample,
            str(tiny_kokoro_ov_path),
            input_prompt,
            tiny_kokoro_speaker_embedding_file_path,
            "--language",
            "en-us",
        ]
        cpp_result = run_sample(cpp_command)

        # Run Python sample with the same Kokoro assets.
        py_script = SAMPLES_PY_DIR / "speech_generation/text2speech.py"
        py_command = [
            sys.executable,
            py_script,
            str(tiny_kokoro_ov_path),
            input_prompt,
            "--speaker_embedding_file_path",
            tiny_kokoro_speaker_embedding_file_path,
            "--language",
            "en-us",
        ]
        py_result = run_sample(py_command)

        assert "Text successfully converted to audio file" in cpp_result.stdout, (
            "C++ Kokoro text2speech sample must be successfully completed"
        )
        assert "Text successfully converted to audio file" in py_result.stdout, (
            "Python Kokoro text2speech sample must be successfully completed"
        )

    @pytest.mark.speech_generation
    @pytest.mark.samples
    def test_sample_kokoro_phonemize_fallback(
        self,
        tiny_kokoro_ov_path: Path,
        tiny_g2p_ov_path: Path,
        tiny_kokoro_speaker_embedding_file_path: str,
    ):
        fallback_prompt = "Vellorin traded copperchimes for rainmint at Candlehaven."

        # Run dedicated C++ fallback sample.
        cpp_sample = SAMPLES_CPP_DIR / "kokoro_phonemize_fallback"
        cpp_command = [
            cpp_sample,
            str(tiny_kokoro_ov_path),
            fallback_prompt,
            "--speaker_embedding_file_path",
            tiny_kokoro_speaker_embedding_file_path,
            "--language",
            "en-us",
            "--phonemize_fallback_model_dir",
            str(tiny_g2p_ov_path),
        ]
        cpp_result = run_sample(cpp_command)

        # Run dedicated Python fallback sample.
        py_script = SAMPLES_PY_DIR / "speech_generation/kokoro_phonemize_fallback.py"
        py_command = [
            sys.executable,
            py_script,
            str(tiny_kokoro_ov_path),
            fallback_prompt,
            "--speaker_embedding_file_path",
            tiny_kokoro_speaker_embedding_file_path,
            "--language",
            "en-us",
            "--phonemize_fallback_model_dir",
            str(tiny_g2p_ov_path),
        ]
        py_result = run_sample(py_command)

        assert "[Info] Saved:" in cpp_result.stdout, "C++ Kokoro fallback sample must save output WAV"
        assert "Phonemize fallback: OpenVINO model" in cpp_result.stdout, (
            "C++ Kokoro fallback sample should report OpenVINO fallback mode"
        )
        assert "OpenVINO fallback" in py_result.stdout, (
            "Python Kokoro fallback sample should report OpenVINO fallback mode"
        )

    @pytest.mark.speech_generation
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["tiny-random-SpeechT5ForTextToSpeech"], indirect=True)
    @pytest.mark.parametrize("input_prompt", ["Test text to speech without speaker embedding file"])
    def test_sample_text_to_speech_no_speaker_embedding_file(self, convert_model, input_prompt):
        # Run C++ sample
        # Example: text2speech spt5_model_dir "Hello everyone" --speaker_embedding_file_path xvector.bin
        cpp_sample = SAMPLES_CPP_DIR / 'text2speech'
        cpp_command = [cpp_sample, convert_model, input_prompt]
        cpp_result = run_sample(cpp_command)

        # Run Python sample
        py_script = SAMPLES_PY_DIR / "speech_generation/text2speech.py"
        py_command = [sys.executable, py_script, convert_model, input_prompt]
        py_result = run_sample(py_command)

        # Run JS sample
        js_script = SAMPLES_JS_DIR / "speech_generation/text2speech.js"
        js_command = ["node", js_script, convert_model, input_prompt]
        js_result = run_sample(js_command)

        assert "Text successfully converted to audio file" in cpp_result.stdout, (
            "C++ sample text2speech must be successfully completed"
        )
        assert "Text successfully converted to audio file" in py_result.stdout, (
            "Python sample text2speech must be successfully completed"
        )
        assert "Text successfully converted to audio file" in js_result.stdout, (
            "JS sample text2speech must be successfully completed"
        )
