# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer, models

from whowhatbench.whowhat_metrics import evaluate_similarity


@pytest.mark.parametrize(
    ("answers_gold", "answers_prediction", "expected"),
    [
        (["cat", "cat"], ["cat", "dog"], [1.0, 0.0]),
        ([["cat", "dog"]], [["cat", "dog"]], [1.0]),
    ],
)
def test_sentence_transformer_similarity(answers_gold, answers_prediction, expected):
    model = SentenceTransformer(modules=[models.BoW(vocab=["cat", "dog"])], device="cpu")

    metrics, details = evaluate_similarity(
        model,
        pd.DataFrame({"answers": answers_gold}),
        pd.DataFrame({"answers": answers_prediction}),
    )

    assert metrics["similarity"] == pytest.approx(sum(expected) / len(expected))
    assert details["similarity"] == pytest.approx(expected)
