# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
import torch
from whowhatbench.whowhat_metrics import evaluate_divergency

_VOCAB = {"zero": 0, "Yes": 9454, "No": 2753}


class _WordTokenizer:
    """Maps each whitespace-separated word to a fixed id; encode() returns a (1, N) tensor like HF tokenizers."""

    def encode(self, text, return_tensors=None):
        return torch.tensor([[_VOCAB[w] for w in text.split()]], dtype=torch.long)


def _divergency(gold, prediction):
    return evaluate_divergency(
        _WordTokenizer(), pd.DataFrame({"answers": [gold]}), pd.DataFrame({"answers": [prediction]})
    )


@pytest.mark.parametrize(
    "gold, prediction, expected_sdt",
    [
        ("Yes", "", 1),  # one-token reference, empty prediction
        ("", "Yes", 1),  # empty reference, one-token prediction
        ("Yes", "No", 1),
        ("Yes", "Yes", 0),
    ],
)
def test_divergency_handles_single_token_answers(gold, prediction, expected_sdt):
    _, per_question = _divergency(gold, prediction)
    assert list(per_question["SDT"]) == [expected_sdt]


def test_divergency_does_not_treat_token_id_zero_as_empty():
    # A single token with id 0 is a real answer, so it matches itself instead of counting as empty.
    _, per_question = _divergency("zero", "zero")
    assert list(per_question["FDT"]) == [1]
    assert list(per_question["SDT"]) == [0]
