# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for generic visual-text loader/dataset helpers.

These cover two fixes that are model-agnostic:
  * ensure_tied_output_embeddings - repairs a randomly-initialized output head
    when only a nested text config declares tie_word_embeddings=True.
  * _load_visual_text_csv - lets --dataset accept a local CSV with
    prompts/images/videos columns for visual-text model types.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image

from whowhatbench.model_loaders import ensure_tied_output_embeddings
from whowhatbench.wwb import _load_visual_text_csv


class _TinyModel(nn.Module):
    def __init__(self, top_tie, nested_tie, vocab=8, dim=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        # Make the head clearly distinct from the embeddings.
        with torch.no_grad():
            self.lm_head.weight.fill_(0.123)
        self.config = SimpleNamespace(
            tie_word_embeddings=top_tie,
            text_config=SimpleNamespace(tie_word_embeddings=nested_tie),
        )

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def tie_weights(self):
        self.lm_head.weight = self.embed.weight


def test_ties_when_nested_config_requests_it():
    model = _TinyModel(top_tie=False, nested_tie=True)
    assert model.get_output_embeddings().weight.data_ptr() != model.get_input_embeddings().weight.data_ptr()
    ensure_tied_output_embeddings(model)
    assert model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()


def test_no_tie_when_top_config_disables_and_nested_false():
    model = _TinyModel(top_tie=False, nested_tie=False)
    before = model.get_output_embeddings().weight.data_ptr()
    ensure_tied_output_embeddings(model)
    # Untouched: no config asked for tying.
    assert model.get_output_embeddings().weight.data_ptr() == before
    assert before != model.get_input_embeddings().weight.data_ptr()


def test_load_visual_text_csv(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(3):
        Image.new("RGB", (16, 16), (i * 10, 0, 0)).save(img_dir / f"{i}.png")

    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text(
        "prompts,images,videos\n"
        "Extract the text.,images/0.png,\n"
        "Extract the text.,images/1.png,\n"
        "Extract the text.,images/2.png,\n"
    )

    args = SimpleNamespace(model_type="visual-text", num_samples=2)
    data = _load_visual_text_csv(args, csv_path)

    assert set(data.keys()) == {"prompts", "images", "videos"}
    # num_samples limits the rows.
    assert len(data["prompts"]) == 2
    assert all(isinstance(im, Image.Image) and im.mode == "RGB" for im in data["images"])
    assert data["videos"] == [None, None]
