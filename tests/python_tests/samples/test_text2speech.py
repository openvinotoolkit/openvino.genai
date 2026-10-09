# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, SAMPLES_C_DIR, SAMPLES_JS_DIR
from test_utils import run_sample
from utils.constants import get_ov_cache_converted_models_dir
from utils.kokoro_test_assets import prepare_tiny_g2p_model_path
from utils.kokoro_test_assets import prepare_tiny_g2p_ov_path
from utils.kokoro_test_assets import prepare_tiny_kokoro_model_path
from utils.kokoro_test_assets import prepare_tiny_kokoro_ov_path

rng = np.random.default_rng(34231)


def compare_speech_audio(cpp_output: Path, python_output: Path, c_output: Path, sample_rate: int, deterministic: bool):
    waveforms = {}
    for name, output in (("C++", cpp_output), ("Python", python_output), ("C", c_output)):
        audio, rate = sf.read(output, dtype="float32")
        assert rate == sample_rate, f"{name} sample rate differs from {sample_rate}"
        assert audio.ndim == 1 and audio.size > 0, f"{name} must produce nonempty mono audio"
        assert np.max(np.abs(audio)) > 1e-4, f"{name} must produce non-silent audio"
        waveforms[name] = audio

    for first, second in (("C++", "Python"), ("C++", "C"), ("Python", "C")):
        first_audio, second_audio = waveforms[first], waveforms[second]
        if deterministic:
            assert first_audio.shape == second_audio.shape, f"{first} and {second} audio lengths differ"
            # The C++ sample writes float WAV data; the C and Python samples clip to PCM16.
            np.testing.assert_allclose(
                np.clip(first_audio, -1, 1), np.clip(second_audio, -1, 1), rtol=0, atol=2 / 32768
            )
        else:
            # SpeechT5 waveforms and lengths vary between independent runs.
            assert 0.5 <= first_audio.size / second_audio.size <= 2.0, (
                f"{first} and {second} audio durations differ substantially"
            )


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
    def test_sample_text_to_speech(self, convert_model, input_prompt, tmp_path: Path):
        # Example: text2speech spt5_model_dir "Hello everyone" --speaker_embedding_file_path xvector.bin
        # Run C++ sample
        cpp_sample = SAMPLES_CPP_DIR / 'text2speech'
        cpp_command = [cpp_sample, convert_model, input_prompt, self.temp_speaker_embedding_file.name]
        cpp_dir = tmp_path / "cpp"
        cpp_dir.mkdir()
        cpp_result = run_sample(cpp_command, cwd=str(cpp_dir))

        # Run Python sample
        py_script = SAMPLES_PY_DIR / "speech_generation/text2speech.py"
        py_command = [sys.executable, py_script, convert_model, input_prompt,
                      "--speaker_embedding_file_path", self.temp_speaker_embedding_file.name]
        python_dir = tmp_path / "python"
        python_dir.mkdir()
        py_result = run_sample(py_command, cwd=str(python_dir))

        # Run C sample
        c_output = tmp_path / "c_output_audio.wav"
        c_command = [
            SAMPLES_C_DIR / "text2speech_c",
            convert_model,
            input_prompt,
            self.temp_speaker_embedding_file.name,
            c_output,
        ]
        run_sample(c_command)
        compare_speech_audio(cpp_dir / "output_audio.wav", python_dir / "output_audio.wav", c_output, 16000, False)

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
        tmp_path: Path,
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
        cpp_dir = tmp_path / "cpp"
        cpp_dir.mkdir()
        cpp_result = run_sample(cpp_command, cwd=str(cpp_dir))

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
        python_dir = tmp_path / "python"
        python_dir.mkdir()
        py_result = run_sample(py_command, cwd=str(python_dir))

        # Run C sample
        c_output = tmp_path / "c_output_audio.wav"
        c_command = [
            SAMPLES_C_DIR / "text2speech_c",
            str(tiny_kokoro_ov_path),
            input_prompt,
            tiny_kokoro_speaker_embedding_file_path,
            c_output,
        ]
        run_sample(c_command)
        compare_speech_audio(cpp_dir / "output_audio.wav", python_dir / "output_audio.wav", c_output, 24000, True)

        assert "Text successfully converted to audio file" in cpp_result.stdout, (
            "C++ Kokoro text2speech sample must be successfully completed"
        )
        assert "Text successfully converted to audio file" in py_result.stdout, (
            "Python Kokoro text2speech sample must be successfully completed"
        )

    @pytest.mark.speech_generation
    @pytest.mark.samples
    def test_c_batch_text_to_speech_kokoro(
        self,
        tiny_kokoro_ov_path: Path,
        tiny_kokoro_speaker_embedding_file_path: str,
        tmp_path: Path,
    ):
        outputs = [tmp_path / "first.wav", tmp_path / "second.wav"]
        command = [
            SAMPLES_C_DIR / "text2speech_c",
            "--batch",
            str(tiny_kokoro_ov_path),
            "Hello from OpenVINO GenAI.",
            "This is a second speech sample.",
            tiny_kokoro_speaker_embedding_file_path,
            *outputs,
            "--speed",
            "1.1",
        ]
        result = run_sample(command)
        assert "Generated 2 speech waveform(s)" in result.stdout
        assert "Applied speech speed: 1.1" in result.stdout
        assert "Metrics: " in result.stdout

        waveforms = []
        for output in outputs:
            audio, rate = sf.read(output, dtype="float32")
            assert rate == 24000
            assert audio.ndim == 1 and audio.size > 0
            assert np.all(np.isfinite(audio))
            assert np.max(np.abs(audio)) > 1e-4
            waveforms.append(audio)
        assert waveforms[0].shape != waveforms[1].shape or not np.array_equal(*waveforms)

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
    def test_sample_text_to_speech_no_speaker_embedding_file(self, convert_model, input_prompt, tmp_path: Path):
        # Run C++ sample
        # Example: text2speech spt5_model_dir "Hello everyone" --speaker_embedding_file_path xvector.bin
        cpp_sample = SAMPLES_CPP_DIR / 'text2speech'
        cpp_command = [cpp_sample, convert_model, input_prompt]
        cpp_dir = tmp_path / "cpp"
        cpp_dir.mkdir()
        cpp_result = run_sample(cpp_command, cwd=str(cpp_dir))

        # Run Python sample
        py_script = SAMPLES_PY_DIR / "speech_generation/text2speech.py"
        py_command = [sys.executable, py_script, convert_model, input_prompt]
        python_dir = tmp_path / "python"
        python_dir.mkdir()
        py_result = run_sample(py_command, cwd=str(python_dir))

        # Run C sample
        c_output = tmp_path / "c_output_audio.wav"
        c_command = [SAMPLES_C_DIR / "text2speech_c", convert_model, input_prompt, "-", c_output]
        run_sample(c_command)
        compare_speech_audio(cpp_dir / "output_audio.wav", python_dir / "output_audio.wav", c_output, 16000, False)

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
