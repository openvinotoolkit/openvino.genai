# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import types
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

from whowhatbench.whowhat_metrics import TranscriptSimilarity
from whowhatbench.utils import to_mono_16k
from whowhatbench.speech_recognition_evaluator import (
    ASRGenAITranscriber,
    ASRGenAIPipelineTranscriber,
    FunASROptimumTranscriber,
    Qwen3ASROptimumTranscriber,
    SpeechRecognitionEvaluator,
)


def _frame(answers):
    return pd.DataFrame({"prompts": [str(i) for i in range(len(answers))], "answers": answers})


def _csv(tmp_path, name, prompts, answers):
    path = tmp_path / name
    pd.DataFrame({"prompts": prompts, "answers": answers}).to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize(
    "language, references, hypotheses, corpus, per_prompt",
    [
        # 1 insertion over 6 reference words: corpus similarity is 5/6, not the 0.75 sample mean.
        ("en", ["the quick brown fox", "hello world"], ["the quick brown fox", "hello there world"], 5 / 6, [1.0, 0.5]),
        ("en", ["hello world"], ["hello world"], 1.0, [1.0]),
        # 3 insertions over 1 reference word: 1 - WER is clamped to 0.
        ("en", ["hello"], ["hello spurious extra words"], 0.0, [0.0]),
        # 1 substitution over 8 reference characters: corpus similarity is 7/8.
        ("zh", ["今天天气很好", "你好"], ["今天天气真好", "你好"], 7 / 8, [5 / 6, 1.0]),
        ("Japanese", ["あいうえお", "かきく"], ["あいうえか", "かきく"], 7 / 8, [4 / 5, 1.0]),
    ],
)
def test_transcript_similarity(language, references, hypotheses, corpus, per_prompt):
    aggregate, per_sample = TranscriptSimilarity(language).evaluate(_frame(references), _frame(hypotheses))
    assert per_sample == {"similarity": per_prompt}
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


def test_score_uses_character_similarity_for_chinese(tmp_path):
    gt = _csv(tmp_path, "gt.csv", ["a"], ["今天天气很好"])
    target = _csv(tmp_path, "target.csv", ["a"], ["今天天气真好"])
    evaluator = SpeechRecognitionEvaluator(gt_data=gt, speech_language="Chinese")

    _, aggregate = evaluator.score(target)
    assert math.isclose(aggregate["similarity"].iloc[0], 5 / 6)


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
        # 1 substitution over 4 reference words.
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
    "export, eos_token_id, pad_token_id, language, decoded",
    [
        ("split", [151643, 151645], 151643, "", "language English<asr_text>decoded"),
        ("legacy", None, None, "", "language English<asr_text>decoded"),
        ("legacy", [151643, 151645], 151643, "English", "decoded"),
    ],
)
def test_qwen3_asr_optimum_transcribes_with_export_settings(export, eos_token_id, pad_token_id, language, decoded):
    model = _FakeSpeechSeq2Seq(torch.tensor([[1, 2, 3, 7, 8]]))
    model.generation_config = types.SimpleNamespace(eos_token_id=eos_token_id, pad_token_id=pad_token_id)
    if export == "split":
        model.audio_encoder = object()
    else:
        model.config = types.SimpleNamespace(decoder_start_token_id=0)
    processor = Mock()
    processor.apply_chat_template.return_value = "audio prompt"
    processor.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "input_features": torch.zeros(1, 128, 100),
        "feature_attention_mask": torch.ones(1, 100),
    }
    processor.tokenizer.pad_token_id = 151643
    processor.tokenizer.eos_token_id = 151645
    processor.batch_decode.return_value = [decoded]
    transcriber = Qwen3ASROptimumTranscriber.__new__(Qwen3ASROptimumTranscriber)
    transcriber.model, transcriber.processor, transcriber.language = model, processor, language
    audio = np.zeros(160, dtype=np.float32)

    assert transcriber.transcribe(audio, 32) == "decoded"
    assert processor.call_args.kwargs == {
        "text": "audio prompt" + (f"language {language}<asr_text>" if language else ""),
        "audio": audio,
        "sampling_rate": 16000,
        "return_tensors": "pt",
    }
    assert model.generate_call["max_new_tokens"] == 32
    assert model.generate_call["do_sample"] is False
    if export == "split":
        assert model.generate_call["input_ids"] is processor.return_value["input_ids"]
        assert "eos_token_id" not in model.generate_call
        assert "pad_token_id" not in model.generate_call
    else:
        assert model.generate_call["decoder_input_ids"] is processor.return_value["input_ids"]
        assert model.generate_call["attention_mask"] is processor.return_value["feature_attention_mask"]
        assert model.generate_call["decoder_start_token_id"] == 0
        if eos_token_id is None:
            assert model.generate_call["eos_token_id"] == [151643, 151645]
            assert model.generate_call["pad_token_id"] == 151643
        else:
            assert "eos_token_id" not in model.generate_call
            assert "pad_token_id" not in model.generate_call
    assert processor.batch_decode.call_args.args[0].tolist() == [[7, 8]]
    assert processor.batch_decode.call_args.kwargs == {"skip_special_tokens": True}


@pytest.mark.parametrize(
    "model_type, language, expected_language",
    [
        ("qwen3_asr", "", None),
        ("qwen3_asr", "Chinese", "Chinese"),
        ("funasr", "", "en"),
    ],
)
def test_asr_genai_uses_asr_pipeline(monkeypatch, model_type, language, expected_language):
    import openvino_genai
    from whowhatbench import speech_recognition_evaluator

    pipeline = Mock()
    pipeline.generate.return_value = types.SimpleNamespace(texts=[" decoded "])
    pipeline_factory = Mock(return_value=pipeline)
    monkeypatch.setattr(openvino_genai, "ASRPipeline", pipeline_factory)
    monkeypatch.setattr(speech_recognition_evaluator, "_get_asr_model_type", lambda _: model_type)

    transcriber = ASRGenAITranscriber.create("model_dir", "CPU", language=language)
    assert isinstance(transcriber, ASRGenAIPipelineTranscriber)
    assert pipeline_factory.call_args.args == ("model_dir", "CPU")

    audio = np.zeros(160, dtype=np.float32)
    assert transcriber.transcribe(audio, 32) == "decoded"
    assert pipeline.generate.call_args.args == (audio.tolist(),)
    expected_kwargs = {"max_new_tokens": 32}
    if expected_language:
        expected_kwargs["language"] = expected_language
    assert pipeline.generate.call_args.kwargs == expected_kwargs


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
