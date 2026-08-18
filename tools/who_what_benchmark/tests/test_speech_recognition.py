# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import math
import types

import numpy as np
import pandas as pd
import pytest
import torch

from whowhatbench import model_loaders
from whowhatbench.whowhat_metrics import WordSimilarity
from whowhatbench.wwb import to_mono_16k
from whowhatbench.speech_recognition_evaluator import (
    FunASROptimumTranscriber,
    SpeechRecognitionEvaluator,
)


def _frame(answers):
    return pd.DataFrame({"prompts": [str(i) for i in range(len(answers))], "answers": answers})


def _csv(tmp_path, name, prompts, answers):
    path = tmp_path / name
    pd.DataFrame({"prompts": prompts, "answers": answers}).to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize(
    "references, hypotheses, corpus, per_prompt",
    [
        # 1 insertion over 6 reference words: the corpus value is 5/6, not the 0.75 mean of the two.
        (["the quick brown fox", "hello world"], ["the quick brown fox", "hello there world"], 5 / 6, [1.0, 0.5]),
        (["hello world"], ["hello world"], 1.0, [1.0]),
        # 3 insertions over 1 reference word: 1 - WER is negative and must clamp to 0.
        (["hello", ""], ["hello", "spurious extra words"], 0.0, [1.0, 0.0]),
        ([], [], 1.0, []),
    ],
)
def test_word_similarity(references, hypotheses, corpus, per_prompt):
    aggregate, per_utterance = WordSimilarity().evaluate(_frame(references), _frame(hypotheses))
    assert per_utterance == {"similarity": per_prompt}
    assert math.isclose(aggregate["similarity"], corpus)


def test_score_reports_similarity(tmp_path):
    gt = _csv(tmp_path, "gt.csv", ["a", "b"], ["the quick brown fox", "hello world"])
    target = _csv(tmp_path, "target.csv", ["a", "b"], ["the quick brown fox", "hello there world"])
    evaluator = SpeechRecognitionEvaluator(gt_data=gt)

    per_prompt, aggregate = evaluator.score(target)
    assert per_prompt.columns.tolist() == ["similarity"]
    assert per_prompt["similarity"].tolist() == [1.0, 0.5]
    assert math.isclose(aggregate["similarity"].iloc[0], 5 / 6)
    assert [example["prompt"] for example in evaluator.worst_examples(top_k=1)] == ["b"]


@pytest.mark.parametrize(
    "columns, error",
    [
        ({"prompts": ["a", "b"]}, "missing required column"),
        ({"prompts": ["a"], "answers": ["x"]}, "differ in length"),
        ({"prompts": ["a", "c"], "answers": ["x", "y"]}, "do not match"),
    ],
)
def test_score_rejects_invalid_predictions(tmp_path, columns, error):
    evaluator = SpeechRecognitionEvaluator(gt_data=_csv(tmp_path, "gt.csv", ["a", "b"], ["x", "y"]))
    target = tmp_path / "target.csv"
    pd.DataFrame(columns).to_csv(target, index=False)

    with pytest.raises(ValueError, match=error):
        evaluator.score(str(target))


class _FakeTranscriber:
    def __init__(self, transcripts):
        self.transcripts = list(transcripts)
        self.calls = []

    def transcribe(self, audio, max_new_tokens):
        self.calls.append((len(audio), max_new_tokens))
        return self.transcripts[len(self.calls) - 1]


AUDIO_DATA = {"prompts": ["a", "b"], "audio": [np.zeros(4, dtype=np.float32), np.zeros(8, dtype=np.float32)]}


@pytest.mark.parametrize(
    "num_samples, calls, prompts, corpus",
    [
        # 1 substitution over 4 reference words
        (None, [(4, 42), (8, 42)], ["a", "b"], 0.75),
        (1, [(4, 42)], ["a"], 1.0),
    ],
)
def test_evaluator_delegates_to_transcriber(num_samples, calls, prompts, corpus):
    base = _FakeTranscriber(["hello world", "second one"])
    evaluator = SpeechRecognitionEvaluator(
        base_model=base, test_data=AUDIO_DATA, max_new_tokens=42, num_samples=num_samples
    )
    assert base.calls == calls
    assert list(evaluator.gt_data["prompts"]) == prompts
    assert list(evaluator.gt_data["answers"]) == ["hello world", "second one"][: len(prompts)]

    target = _FakeTranscriber(["hello world", "second two"])
    _, aggregate = evaluator.score(target)
    assert target.calls == calls
    assert math.isclose(aggregate["similarity"].iloc[0], corpus)


@pytest.mark.parametrize(
    "files, kind",
    [
        ({"configuration.json": {"framework": "pytorch", "model": {"type": "funasr"}}}, "source"),
        ({"config.json": {"model_type": "fun_asr"}}, "export"),
        ({"config.json": {"model_type": "gemma4_unified"}}, None),
    ],
)
def test_funasr_model_kind_local(tmp_path, files, kind):
    for name, payload in files.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    assert model_loaders.funasr_model_kind(str(tmp_path)) == kind


class _FakeSpeechSeq2Seq:
    def __init__(self, generated):
        self.generated = generated
        self.preprocess_call = None
        self.generate_call = None

    def preprocess_input(self, waveform, sampling_rate, **kwargs):
        self.preprocess_call = {"sampling_rate": sampling_rate, **kwargs}
        return {"input_features": torch.zeros(1, 2, 3), "decoder_input_ids": torch.tensor([[1, 2, 3]])}

    def generate(self, **kwargs):
        self.generate_call = kwargs
        return self.generated


class _FakeTokenizer:
    def __init__(self):
        self.decoded = None

    def batch_decode(self, ids, **kwargs):
        self.decoded = ids
        return [" decoded "]


@pytest.mark.parametrize(
    "generated",
    [torch.tensor([[1, 2, 3, 7, 8]]), types.SimpleNamespace(sequences=torch.tensor([[1, 2, 3, 7, 8]]))],
)
def test_funasr_optimum_transcriber_decodes_generated_ids_only(generated):
    model, tokenizer = _FakeSpeechSeq2Seq(generated), _FakeTokenizer()

    assert FunASROptimumTranscriber(model, tokenizer, "en").transcribe(np.zeros(16, dtype=np.float32), 32) == "decoded"
    assert model.preprocess_call == {"sampling_rate": 16000, "language": "en"}
    assert model.generate_call["max_new_tokens"] == 32
    # the 3 prompt ids are dropped, only the generated ids are decoded
    assert tokenizer.decoded.tolist() == [[7, 8]]


@pytest.mark.parametrize(
    "audio, peak",
    [
        (np.stack([np.ones(4800), -np.ones(4800)], axis=1), 0.0),  # the two channels cancel out
        (np.sin(np.arange(4800, dtype=np.float64) / 100.0), 1.0),
    ],
)
def test_to_mono_16k(audio, peak):
    resampled = to_mono_16k(audio, 48000)
    assert resampled.shape == (1600,)
    assert resampled.dtype == np.float32
    assert math.isclose(np.abs(resampled).max(), peak, abs_tol=1e-3)
