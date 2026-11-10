# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from whowhatbench.text_evaluator import TextEvaluator


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
