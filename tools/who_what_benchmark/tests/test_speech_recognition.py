# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

import pandas as pd
import pytest

from whowhatbench.whowhat_metrics import WordErrorRate
from whowhatbench.speech_recognition_evaluator import SpeechRecognitionEvaluator


def _frame(prompts, answers):
    return pd.DataFrame({"prompts": prompts, "answers": answers})


def _csv(tmp_path, name, prompts, answers):
    path = tmp_path / name
    _frame(prompts, answers).to_csv(path, index=False)
    return str(path)


def test_wer_corpus_and_per_utterance():
    gt = _frame(["a", "b"], ["the quick brown fox", "hello world"])
    pred = _frame(["a", "b"], ["the quick brown fox", "hello there world"])
    aggregate, per_prompt = WordErrorRate().evaluate(gt, pred)
    # 1 insertion over 6 reference words -> corpus 1/6; mean-utterance would be 0.25.
    assert per_prompt["WER"] == [0.0, 0.5]
    assert math.isclose(aggregate["WER"], 1 / 6)


def test_wer_identical_is_zero():
    gt = _frame(["a"], ["hello world"])
    assert WordErrorRate().evaluate(gt, _frame(["a"], ["hello world"]))[0]["WER"] == 0.0


def test_wer_empty_reference_counts_insertions():
    gt = _frame(["a", "b"], ["hello", ""])
    pred = _frame(["a", "b"], ["hello", "spurious words"])
    aggregate, per_prompt = WordErrorRate().evaluate(gt, pred)
    assert aggregate["WER"] == 2.0
    assert per_prompt["WER"] == [0.0, 2.0]


def test_wer_empty_data_is_zero():
    aggregate, per_prompt = WordErrorRate().evaluate(_frame([], []), _frame([], []))
    assert aggregate["WER"] == 0.0
    assert per_prompt["WER"] == []


def test_wer_length_mismatch_raises():
    with pytest.raises(ValueError, match="counts differ"):
        WordErrorRate().evaluate(_frame(["a", "b"], ["x", "y"]), _frame(["a"], ["x"]))


def test_score_row_count_mismatch(tmp_path):
    evaluator = SpeechRecognitionEvaluator(gt_data=_csv(tmp_path, "gt.csv", ["a", "b"], ["x", "y"]))
    with pytest.raises(ValueError, match="differ in length"):
        evaluator.score(_csv(tmp_path, "target.csv", ["a"], ["x"]))


def test_score_missing_column(tmp_path):
    evaluator = SpeechRecognitionEvaluator(gt_data=_csv(tmp_path, "gt.csv", ["a"], ["x"]))
    bad = tmp_path / "target.csv"
    pd.DataFrame({"prompts": ["a"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        evaluator.score(str(bad))


def test_score_prompt_ids_mismatch(tmp_path):
    evaluator = SpeechRecognitionEvaluator(gt_data=_csv(tmp_path, "gt.csv", ["a", "b"], ["x", "y"]))
    with pytest.raises(ValueError, match="do not match"):
        evaluator.score(_csv(tmp_path, "target.csv", ["a", "c"], ["x", "y"]))


def test_score_end_to_end(tmp_path):
    gt = _csv(tmp_path, "gt.csv", ["a", "b"], ["the quick brown fox", "hello world"])
    target = _csv(tmp_path, "target.csv", ["a", "b"], ["the quick brown fox", "hello there world"])
    evaluator = SpeechRecognitionEvaluator(gt_data=gt)
    _, aggregate = evaluator.score(target)
    assert math.isclose(aggregate["WER"].iloc[0], 1 / 6)
