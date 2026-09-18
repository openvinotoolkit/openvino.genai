# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess  # nosec B404
import sys
import tempfile
import wave
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


@pytest.fixture(scope="module")
def qwen3_reference_audio_wav_path(tmp_path_factory) -> str:
    samples = 24_000
    timeline = np.arange(samples, dtype=np.float32) / np.float32(samples)
    waveform = 0.12 * np.sin(2.0 * np.pi * 220.0 * timeline) + 0.05 * np.sin(2.0 * np.pi * 440.0 * timeline + 0.4)
    output_dir = tmp_path_factory.mktemp("qwen3_tts_ref_audio")
    wav_path = output_dir / "reference_24k.wav"
    pcm = np.asarray(np.clip(waveform, -1.0, 1.0) * 32767.0, dtype=np.int16)
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(pcm.tobytes())
    return str(wav_path)


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

    @pytest.mark.speech_generation
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model",
        ["tiny-random-qwen3-tts"],
        indirect=True,
    )
    def test_sample_qwen3_tts_base_xvector_and_icl_save_reuse(
        self,
        convert_model,
        qwen3_reference_audio_wav_path: str,
        tmp_path: Path,
    ):
        prompt = "Hello from OpenVINO GenAI Qwen3 Base sample."
        ref_text = "This is a tiny random reference transcript for ICL reuse testing."

        run_dir = tmp_path / "qwen3_tts_base_sample"
        run_dir.mkdir(parents=True, exist_ok=True)
        speaker_embedding_path = run_dir / "speaker_embedding.bin"
        ref_codes_path = run_dir / "ref_codes.bin"

        cpp_xvector_wav = run_dir / "cpp_xvector.wav"
        py_xvector_wav = run_dir / "py_xvector.wav"
        cpp_xvector_reuse_wav = run_dir / "cpp_xvector_reuse.wav"
        py_xvector_reuse_wav = run_dir / "py_xvector_reuse.wav"
        cpp_icl_wav = run_dir / "cpp_icl.wav"
        py_icl_wav = run_dir / "py_icl.wav"
        cpp_icl_reuse_wav = run_dir / "cpp_icl_reuse.wav"
        py_icl_reuse_wav = run_dir / "py_icl_reuse.wav"
        generated_wav_paths = [
            cpp_xvector_wav,
            py_xvector_wav,
            cpp_xvector_reuse_wav,
            py_xvector_reuse_wav,
            cpp_icl_wav,
            py_icl_wav,
            cpp_icl_reuse_wav,
            py_icl_reuse_wav,
        ]

        cpp_sample = SAMPLES_CPP_DIR / "qwen3_tts"
        py_script = SAMPLES_PY_DIR / "speech_generation/qwen3_tts.py"

        cpp_xvector_first = run_sample(
            [
                cpp_sample,
                "base",
                convert_model,
                prompt,
                "--ref_audio_wav_path",
                qwen3_reference_audio_wav_path,
                "--language",
                "english",
                "--max_new_tokens",
                "32",
                "--output_wav_path",
                str(cpp_xvector_wav),
                "--save_speaker_embedding_file_path",
                str(speaker_embedding_path),
            ]
        )
        py_xvector_first = run_sample(
            [
                sys.executable,
                py_script,
                "base",
                convert_model,
                prompt,
                "--ref_audio_wav_path",
                qwen3_reference_audio_wav_path,
                "--language",
                "english",
                "--max_new_tokens",
                "32",
                "--output_wav_path",
                str(py_xvector_wav),
                "--save_speaker_embedding_file_path",
                str(speaker_embedding_path),
            ]
        )

        cpp_xvector_reuse = run_sample(
            [
                cpp_sample,
                "base",
                convert_model,
                prompt,
                "--speaker_embedding_file_path",
                str(speaker_embedding_path),
                "--language",
                "english",
                "--max_new_tokens",
                "32",
                "--output_wav_path",
                str(cpp_xvector_reuse_wav),
            ]
        )
        py_xvector_reuse = run_sample(
            [
                sys.executable,
                py_script,
                "base",
                convert_model,
                prompt,
                "--speaker_embedding_file_path",
                str(speaker_embedding_path),
                "--language",
                "english",
                "--max_new_tokens",
                "32",
                "--output_wav_path",
                str(py_xvector_reuse_wav),
            ]
        )

        cpp_icl_first = run_sample(
            [
                cpp_sample,
                "base",
                convert_model,
                prompt,
                "--ref_audio_wav_path",
                qwen3_reference_audio_wav_path,
                "--ref_text",
                ref_text,
                "--language",
                "english",
                "--max_new_tokens",
                "32",
                "--output_wav_path",
                str(cpp_icl_wav),
                "--save_speaker_embedding_file_path",
                str(speaker_embedding_path),
                "--save_ref_codec_ids_file_path",
                str(ref_codes_path),
            ]
        )
        py_icl_first = run_sample(
            [
                sys.executable,
                py_script,
                "base",
                convert_model,
                prompt,
                "--ref_audio_wav_path",
                qwen3_reference_audio_wav_path,
                "--ref_text",
                ref_text,
                "--language",
                "english",
                "--max_new_tokens",
                "32",
                "--output_wav_path",
                str(py_icl_wav),
                "--save_speaker_embedding_file_path",
                str(speaker_embedding_path),
                "--save_ref_codec_ids_file_path",
                str(ref_codes_path),
            ]
        )

        cpp_icl_reuse = run_sample(
            [
                cpp_sample,
                "base",
                convert_model,
                prompt,
                "--speaker_embedding_file_path",
                str(speaker_embedding_path),
                "--ref_text",
                ref_text,
                "--ref_codec_ids_file_path",
                str(ref_codes_path),
                "--language",
                "english",
                "--max_new_tokens",
                "32",
                "--output_wav_path",
                str(cpp_icl_reuse_wav),
            ]
        )
        py_icl_reuse = run_sample(
            [
                sys.executable,
                py_script,
                "base",
                convert_model,
                prompt,
                "--speaker_embedding_file_path",
                str(speaker_embedding_path),
                "--ref_text",
                ref_text,
                "--ref_codec_ids_file_path",
                str(ref_codes_path),
                "--language",
                "english",
                "--max_new_tokens",
                "32",
                "--output_wav_path",
                str(py_icl_reuse_wav),
            ]
        )

        for result in [
            cpp_xvector_first,
            py_xvector_first,
            cpp_xvector_reuse,
            py_xvector_reuse,
            cpp_icl_first,
            py_icl_first,
            cpp_icl_reuse,
            py_icl_reuse,
        ]:
            assert "Text successfully converted to audio file" in result.stdout

        for result in [cpp_xvector_first, py_xvector_first, cpp_icl_first, py_icl_first]:
            assert "Saved speaker embedding" in result.stdout

        for result in [cpp_icl_first, py_icl_first]:
            assert "Saved reference codes" in result.stdout

        assert os.path.exists(speaker_embedding_path), "Speaker embedding artifact should exist"
        assert os.path.exists(ref_codes_path), "Reference codes artifact should exist"
        for wav_path in generated_wav_paths:
            assert os.path.exists(wav_path), f"Generated WAV output should exist: {wav_path}"

    @pytest.mark.speech_generation
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model",
        ["tiny-random-qwen3-tts-customvoice"],
        indirect=True,
    )
    def test_sample_qwen3_tts_customvoice(self, convert_model):
        prompt = "Hello from Qwen3 custom voice sample test."
        instruct = "Speak with clear articulation and a friendly, professional tone."

        cpp_sample = SAMPLES_CPP_DIR / "qwen3_tts"
        cpp_command = [
            cpp_sample,
            "customvoice",
            convert_model,
            prompt,
            "--speaker",
            "serena",
            "--language",
            "english",
            "--instruct",
            instruct,
            "--max_new_tokens",
            "32",
        ]
        cpp_result = run_sample(cpp_command)

        py_script = SAMPLES_PY_DIR / "speech_generation/qwen3_tts.py"
        py_command = [
            sys.executable,
            py_script,
            "customvoice",
            convert_model,
            prompt,
            "--speaker",
            "serena",
            "--language",
            "english",
            "--instruct",
            instruct,
            "--max_new_tokens",
            "32",
        ]
        py_result = run_sample(py_command)

        assert "Text successfully converted to audio file" in cpp_result.stdout
        assert "Text successfully converted to audio file" in py_result.stdout

    @pytest.mark.speech_generation
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model",
        ["tiny-random-qwen3-tts-voicedesign"],
        indirect=True,
    )
    def test_sample_qwen3_tts_voice_design(self, convert_model):
        prompt = "Hello from Qwen3 voice design sample test."
        instruct = "Some voice design instruct prompt"

        cpp_sample = SAMPLES_CPP_DIR / "qwen3_tts"
        cpp_command = [
            cpp_sample,
            "voice-design",
            convert_model,
            prompt,
            "--language",
            "english",
            "--instruct",
            instruct,
            "--max_new_tokens",
            "32",
        ]
        cpp_result = run_sample(cpp_command)

        py_script = SAMPLES_PY_DIR / "speech_generation/qwen3_tts.py"
        py_command = [
            sys.executable,
            py_script,
            "voice-design",
            convert_model,
            prompt,
            "--language",
            "english",
            "--instruct",
            instruct,
            "--max_new_tokens",
            "32",
        ]
        py_result = run_sample(py_command)

        assert "Text successfully converted to audio file" in cpp_result.stdout
        assert "Text successfully converted to audio file" in py_result.stdout
