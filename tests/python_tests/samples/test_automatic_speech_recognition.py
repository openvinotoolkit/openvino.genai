# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, SAMPLES_C_DIR, SAMPLES_JS_DIR
from test_utils import run_sample


class TestAutomaticSpeechRecognition:
    @pytest.mark.automatic_speech_recognition
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["WhisperTiny"], indirect=True)
    @pytest.mark.parametrize("download_test_content", ["how_are_you_doing_today.wav"], indirect=True)
    @pytest.mark.parametrize("show_perf", [False, True])
    def test_sample_automatic_speech_recognition(self, convert_model, download_test_content, show_perf):
        # Run C++ sample
        cpp_sample = SAMPLES_CPP_DIR / "automatic_speech_recognition"
        cpp_command = [cpp_sample, convert_model, download_test_content]
        cpp_result = run_sample(cpp_command)

        # Run Python sample
        py_script = SAMPLES_PY_DIR / "automatic_speech_recognition/automatic_speech_recognition.py"
        py_command = [sys.executable, py_script, convert_model, download_test_content]
        if show_perf:
            py_command.append("--perf")
        py_result = run_sample(py_command)

        # Run C sample
        c_sample = SAMPLES_C_DIR / "whisper_speech_recognition_c"
        c_command = [c_sample, convert_model, download_test_content]
        c_result = run_sample(c_command)

        # Run JS sample
        js_sample = SAMPLES_JS_DIR / "automatic_speech_recognition/automatic_speech_recognition.js"
        js_command = ["node", js_sample, convert_model, download_test_content]
        js_result = run_sample(js_command)

        # Compare results
        py_text = py_result.stdout
        if show_perf:
            py_text, separator, metrics = py_text.partition("\n=== Performance Metrics ===\n")
            assert separator
            assert "Encoder inference:" in metrics
            assert "Decoder inference:" in metrics
            metric_prefixes = (
                "Load time:",
                "Total generate duration:",
                "Encoder inference:",
                "Decoder TTFT:",
                "Decoder inference:",
                "TPOT:",
                "Throughput:",
                "Encoder raw calls:",
                "Decoder raw steps:",
            )
            py_text += "".join(
                line for line in metrics.splitlines(keepends=True) if not line.startswith(metric_prefixes)
            )
        else:
            assert "=== Performance Metrics ===" not in py_text

        assert py_text == cpp_result.stdout, "Python and C++ results should match"
        assert py_text == js_result.stdout, "Python and JS results should match"
        assert py_text == c_result.stdout, "Python and C results should match"
