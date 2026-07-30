# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import numpy as np
import pytest


class _Speech:
    def __init__(self, data):
        self.data = data


class _SpeechResult:
    def __init__(self, data, sample_rate=16000):
        self.speeches = [_Speech(data)]
        self.output_sample_rate = sample_rate


class _RngModel:
    """Fake TTS model whose output depends on the global torch RNG."""

    def get_speaker_embedding_shape(self):
        return (1, 512)

    def generate(self, prompt, speaker_embedding=None, **kwargs):
        import torch

        audio = torch.randn(16).cpu().numpy().astype(np.float32)
        return _SpeechResult(audio)


def _make_evaluator(tmp_path, seed, prompts):
    """Build a SpeechGenerationEvaluator from ground-truth CSV (no model load)."""
    import pandas as pd

    from whowhatbench.speech_generation_evaluator import SpeechGenerationEvaluator

    gt_file = tmp_path / "gt.csv"
    pd.DataFrame({"prompts": prompts, "audio": ["x.wav"] * len(prompts)}).to_csv(gt_file, index=False)

    return SpeechGenerationEvaluator(
        gt_data=str(gt_file),
        test_data={"prompts": prompts},
        num_samples=len(prompts),
        seed=seed,
    )


def test_generate_data_seeds_before_each_prompt(tmp_path, monkeypatch):
    """torch.manual_seed(seed) is called once per prompt, before each generation."""
    torch = pytest.importorskip("torch")

    prompts = ["one", "two", "three"]
    evaluator = _make_evaluator(tmp_path, seed=123, prompts=prompts)

    seed_calls = []
    monkeypatch.setattr(torch, "manual_seed", lambda value: seed_calls.append(value))

    evaluator._generate_data(_RngModel(), audio_dir=str(tmp_path / "out"))

    assert seed_calls == [123, 123, 123]


def test_generate_data_deterministic_across_runs(tmp_path):
    """A seeded evaluator produces identical audio across two independent runs."""
    pytest.importorskip("torch")

    prompts = ["hello", "world"]

    def collect():
        evaluator = _make_evaluator(tmp_path, seed=7, prompts=prompts)
        frame = evaluator._generate_data(_RngModel(), audio_dir=str(tmp_path / "run"))
        import soundfile as sf

        return [sf.read(path)[0] for path in frame["audio"].values]

    first = collect()
    second = collect()

    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_generate_data_no_seed_leaves_rng_untouched(tmp_path, monkeypatch):
    """With seed=None the evaluator must not touch the global torch RNG."""
    torch = pytest.importorskip("torch")

    prompts = ["only"]
    evaluator = _make_evaluator(tmp_path, seed=None, prompts=prompts)

    seed_calls = []
    monkeypatch.setattr(torch, "manual_seed", lambda value: seed_calls.append(value))

    evaluator._generate_data(_RngModel(), audio_dir=str(tmp_path / "out"))

    assert seed_calls == []
