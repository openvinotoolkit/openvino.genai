# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def asr_sample(monkeypatch):
    monkeypatch.setitem(sys.modules, "openvino_genai", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace())
    source = (
        Path(__file__).resolve().parents[3]
        / "samples"
        / "python"
        / "automatic_speech_recognition"
        / "automatic_speech_recognition.py"
    )
    spec = importlib.util.spec_from_file_location("asr_sample_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_asr_sample_retains_compilation_cache(asr_sample):
    assert asr_sample.get_config_for_cache() == {"CACHE_DIR": "asr_cache"}


@pytest.mark.parametrize("show_perf", [False, True])
def test_asr_sample_performance_output_is_opt_in(asr_sample, monkeypatch, capsys, show_perf):
    class Result:
        chunks = [[SimpleNamespace(start_ts=0.0, end_ts=1.0, text="transcript")]]
        words = [[SimpleNamespace(start_ts=0.0, end_ts=1.0, text="transcript")]]

        def __str__(self):
            return "transcript"

    pipeline = SimpleNamespace(
        get_generation_config=lambda: SimpleNamespace(),
        generate=lambda audio, config: Result(),
    )
    monkeypatch.setattr(asr_sample.openvino_genai, "ASRPipeline", lambda *args, **kwargs: pipeline, raising=False)
    monkeypatch.setattr(asr_sample, "read_wav", lambda path: [0.0])
    monkeypatch.setattr(asr_sample, "print_perf_metrics", lambda result: print("performance diagnostics"))
    monkeypatch.setattr(sys, "argv", ["asr_sample", "model", "audio.wav"] + (["--perf"] if show_perf else []))

    asr_sample.main()

    output = capsys.readouterr().out
    expected = "transcript\n"
    if show_perf:
        expected += "performance diagnostics\n"
    expected += "timestamps: [0.00, 1.00] text: transcript\n[0.00, 1.00]: transcript\n"
    assert output == expected
