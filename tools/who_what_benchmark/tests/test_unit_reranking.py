# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np


def test_reranking_batch_size_one_processes_documents_individually(monkeypatch, tmp_path):
    """Test that a batch size of one evaluates every document separately."""
    from whowhatbench import reranking_evaluator

    monkeypatch.setattr(
        reranking_evaluator,
        "prepare_default_data",
        lambda num_samples: {"query": ["query"], "passages": [["document 0", "document 1", "document 2"]]},
    )

    evaluator = object.__new__(reranking_evaluator.RerankingEvaluator)
    evaluator.num_samples = 1
    evaluator.tokenizer = object()
    evaluator.batch_size = 1
    model = SimpleNamespace(config=SimpleNamespace(model_type="test"))
    calls = []

    def gen_answer_fn(model, tokenizer, query, documents):
        calls.append((query, documents))
        return np.array([[0, len(calls)]])

    result_data = evaluator._generate_data(model, gen_answer_fn, str(tmp_path))
    scores = np.load(result_data.loc[0, "top_n_scores_path"])

    assert calls == [
        ("query", ["document 0"]),
        ("query", ["document 1"]),
        ("query", ["document 2"]),
    ]
    np.testing.assert_array_equal(scores[:, 0], [0, 1, 2])
    np.testing.assert_array_equal(scores[:, 1], [1, 2, 3])


def test_reranking_uses_embeds_batch_size(monkeypatch, tmp_path):
    """Test that reranking uses embeds_batch_size to split documents."""
    from whowhatbench import reranking_evaluator

    monkeypatch.setattr(
        reranking_evaluator,
        "prepare_default_data",
        lambda num_samples: {"query": ["query"], "passages": [["document 0", "document 1", "document 2"]]},
    )

    evaluator = object.__new__(reranking_evaluator.RerankingEvaluator)
    evaluator.num_samples = 1
    evaluator.tokenizer = object()
    evaluator.batch_size = 2
    model = SimpleNamespace(config=SimpleNamespace(model_type="test"))
    calls = []

    def gen_answer_fn(model, tokenizer, query, documents):
        calls.append(documents)
        return np.array([[index, index + 1] for index in range(len(documents))])

    result_data = evaluator._generate_data(model, gen_answer_fn, str(tmp_path))
    scores = np.load(result_data.loc[0, "top_n_scores_path"])

    assert calls == [["document 0", "document 1"], ["document 2"]]
    np.testing.assert_array_equal(scores[:, 0], [0, 1, 2])


def test_reranking_without_embeds_batch_size_keeps_single_call(monkeypatch, tmp_path):
    """Test that reranking without a batch size keeps the original behavior."""
    from whowhatbench import reranking_evaluator

    monkeypatch.setattr(
        reranking_evaluator,
        "prepare_default_data",
        lambda num_samples: {"query": ["query"], "passages": [["document 0", "document 1"]]},
    )

    evaluator = object.__new__(reranking_evaluator.RerankingEvaluator)
    evaluator.num_samples = 1
    evaluator.tokenizer = object()
    evaluator.batch_size = None
    model = SimpleNamespace(config=SimpleNamespace(model_type="test"))
    calls = []

    def gen_answer_fn(model, tokenizer, query, documents):
        calls.append(documents)
        return np.array([[0, 1], [1, 2]])

    result_data = evaluator._generate_data(model, gen_answer_fn, str(tmp_path))
    scores = np.load(result_data.loc[0, "top_n_scores_path"])

    assert calls == [["document 0", "document 1"]]
    np.testing.assert_array_equal(scores[:, 0], [0, 1])
