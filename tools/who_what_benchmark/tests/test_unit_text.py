# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer, models as tokenizer_models, pre_tokenizers
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from whowhatbench.text_evaluator import TextEvaluator
from whowhatbench.whowhat_metrics import TextSimilarity, evaluate_similarity


def test_text_evaluator_batch_generation_path(monkeypatch):
    monkeypatch.setattr("whowhatbench.text_evaluator.TextSimilarity", lambda _: None)

    class DummyGenerationConfig:
        pass

    class DummyModel:
        def __init__(self):
            self.calls = []

        def generate(self, batch, configs):
            self.calls.append((list(batch), list(configs)))
            return [
                SimpleNamespace(m_generation_ids=["answer1"]),
                SimpleNamespace(m_generation_ids=["answer2"]),
            ]

    prompts = ["1", "2", "3", "4"]
    config = DummyGenerationConfig()
    model = DummyModel()

    evaluator = TextEvaluator(
        base_model=model,
        tokenizer=None,
        test_data=prompts,
        metrics="similarity",
        generation_config=config,
        seqs_per_request=2,
        long_prompt=False,
    )

    assert len(model.calls) == 2
    assert model.calls[0][0] == prompts[:2] and model.calls[1][0] == prompts[2:]
    assert evaluator.gt_data["prompts"].tolist() == prompts
    assert model.calls[0][1] == [config, config]
    assert evaluator.gt_data["answers"].tolist() == ["answer1", "answer2", "answer1", "answer2"]


@pytest.mark.parametrize(
    ("answers_gold", "answers_prediction", "expected", "expected_per_chat"),
    [
        (["cat", "cat"], ["cat", "dog"], [1.0, 0.0], None),
        ([["cat", "dog"]], [["cat", "cat"]], [0.5], [1.0, 0.0]),
    ],
)
def test_text_similarity_scores(answers_gold, answers_prediction, expected, expected_per_chat):
    class DummyEmbeddings:
        def encode(self, texts, convert_to_tensor, show_progress_bar):
            assert convert_to_tensor
            assert not show_progress_bar
            vectors = {"cat": [1.0, 0.0], "dog": [0.0, 1.0]}
            return torch.tensor([vectors[text] for text in texts])

    model = DummyEmbeddings()

    metrics, details = evaluate_similarity(
        model,
        pd.DataFrame({"answers": answers_gold}),
        pd.DataFrame({"answers": answers_prediction}),
    )

    assert metrics["similarity"] == pytest.approx(sum(expected) / len(expected))
    assert details["similarity"] == pytest.approx(expected)
    if expected_per_chat is None:
        assert "similarity_per_chat_replicas" not in details
    else:
        assert len(details["similarity_per_chat_replicas"]) == 1
        assert details["similarity_per_chat_replicas"][0] == pytest.approx(expected_per_chat)


@pytest.mark.parametrize("answers", [["cat", "dog"], [["cat", "dog"]]])
def test_text_similarity_loads_local_transformer(tmp_path, answers):
    vocab = {"[UNK]": 0, "[PAD]": 1, "cat": 2, "dog": 3}
    tokenizer = Tokenizer(tokenizer_models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(tmp_path)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        pad_token_id=vocab["[PAD]"],
    )
    BertModel(config).save_pretrained(tmp_path)

    similarity = TextSimilarity(str(tmp_path))
    data = pd.DataFrame({"answers": answers})
    metrics, details = similarity.evaluate(data, data)

    assert metrics["similarity"] == pytest.approx(1.0)
    assert details["similarity"] == pytest.approx([1.0] * len(answers))
    if isinstance(answers[0], list):
        assert details["similarity_per_chat_replicas"][0] == pytest.approx([1.0, 1.0])
