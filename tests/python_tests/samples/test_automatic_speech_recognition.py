# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import importlib.util
import sys
import re

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, SAMPLES_C_DIR, SAMPLES_JS_DIR
from test_utils import run_sample


def filter_word_level_timestamps(text: str) -> str:
    """
    example:
     How are you doing today?
    timestamps: [0.00, 2.00] text:  How are you doing today?
    [0.00, 0.58]:  How
    [0.58, 0.70]:  are
    [0.70, 0.80]:  you
    [0.80, 1.06]:  doing
    [1.06, 1.40]:  today?
    """
    pattern = r"\[\d+\.\d{2}, \d+\.\d{2}\]:\s+\S+"
    filtered_text = re.sub(pattern, "", text).strip()
    return filtered_text


class TestAutomaticSpeechRecognition:
    @pytest.mark.automatic_speech_recognition
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["WhisperTiny"], indirect=True)
    @pytest.mark.parametrize("download_test_content", ["how_are_you_doing_today.wav"], indirect=True)
    def test_sample_automatic_speech_recognition(self, convert_model, download_test_content):
        # Run C++ sample
        cpp_sample = SAMPLES_CPP_DIR / "automatic_speech_recognition"
        cpp_command = [cpp_sample, convert_model, download_test_content]
        cpp_result = run_sample(cpp_command)

        # Run Python sample
        py_script = SAMPLES_PY_DIR / "automatic_speech_recognition/automatic_speech_recognition.py"
        py_command = [sys.executable, py_script, convert_model, download_test_content]
        py_result = run_sample(py_command)

        # Run C sample
        c_sample = SAMPLES_C_DIR / "whisper_speech_recognition_c"
        c_command = [c_sample, convert_model, download_test_content]
        c_result = run_sample(c_command)

        # Run the generic ASR C sample, including word-level timestamps.
        asr_c_sample = SAMPLES_C_DIR / "automatic_speech_recognition_c"
        asr_c_result = run_sample(
            [
                asr_c_sample,
                convert_model,
                download_test_content,
                "--language",
                "<|en|>",
                "--timestamps",
                "--word-timestamps",
            ]
        )

        # Run JS sample
        js_sample = SAMPLES_JS_DIR / "automatic_speech_recognition/automatic_speech_recognition.js"
        js_command = ["node", js_sample, convert_model, download_test_content]
        js_result = run_sample(js_command)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, "Python and C++ results should match"
        assert py_result.stdout == js_result.stdout, "Python and JS results should match"
        assert py_result.stdout == asr_c_result.stdout, "Python and ASR C results should match"
        # Whisper C API has no word-level timestamps support which enabled in Python and CPP samples
        # ticket to enable C API: 180115
        assert filter_word_level_timestamps(py_result.stdout) == c_result.stdout.strip(), (
            "Python and C results should match without word-level timestamps"
        )

    @pytest.mark.automatic_speech_recognition
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, language",
        [
            ("WhisperTiny", "<|en|>"),
            pytest.param(
                "TinyRandomQwen3ASR",
                "English",
                marks=pytest.mark.skipif(
                    importlib.util.find_spec("qwen_asr") is None, reason="Qwen3-ASR export requires qwen-asr"
                ),
            ),
            pytest.param(
                "TinyRandomFunASR",
                "English",
                marks=pytest.mark.skipif(
                    importlib.util.find_spec("funasr") is None, reason="Fun-ASR export requires funasr"
                ),
            ),
        ],
        indirect=["convert_model"],
    )
    @pytest.mark.parametrize("download_test_content", ["how_are_you_doing_today.wav"], indirect=True)
    def test_asr_c_sample_backends(self, convert_model, language, download_test_content):
        import librosa
        import openvino_genai

        # Bound generation for random fixtures; their output need not contain EOS.
        pipeline = openvino_genai.ASRPipeline(convert_model, "CPU")
        speech, _ = librosa.load(download_test_content, sr=16000)
        config = pipeline.get_generation_config()
        config.language = language
        config.max_new_tokens = 16
        expected = str(pipeline.generate(speech.tolist(), config))
        result = run_sample(
            [
                SAMPLES_C_DIR / "automatic_speech_recognition_c",
                convert_model,
                download_test_content,
                "--language",
                language,
                "--max-new-tokens",
                "16",
            ]
        )
        assert result.stdout.rstrip("\n") == expected, "C and Python ASR results should match"
