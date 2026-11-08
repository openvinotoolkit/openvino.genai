# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from whowhatbench.inputs_preprocessors.molmo2 import Molmo2InputsPreprocessor
from whowhatbench.utils import uniform_frame_indices


@pytest.mark.parametrize(
    "total_num_frames, num_frames, expected",
    [
        # The frames are spread over the whole clip, not taken from its beginning.
        (19, 10, [0, 1, 3, 5, 7, 9, 11, 13, 15, 17]),
        (40, 4, [0, 10, 20, 30]),
        # A shorter video keeps all of its frames.
        (3, 10, [0, 1, 2]),
    ],
)
def test_uniform_frame_indices_cover_the_whole_video(total_num_frames, num_frames, expected):
    metadata = SimpleNamespace(total_num_frames=total_num_frames)
    assert uniform_frame_indices(metadata, num_frames=num_frames).tolist() == expected


@pytest.mark.parametrize(
    "token_type_ids, prefix_len, expected",
    [
        # New image tokens after cached image tokens: the cached image states are computed again.
        ([0, 1, 1, 0, 0, 1, 1, 0], 5, 1),
        # Only text follows the cached image tokens: the cache stays valid.
        ([0, 1, 1, 0, 0, 0, 0], 5, 5),
        # The first image comes with the new tokens: nothing cached attends to it.
        ([0, 0, 0, 1, 1, 0], 3, 3),
    ],
)
def test_molmo2_cache_is_recomputed_from_the_first_image_when_images_are_added(token_type_ids, prefix_len, expected):
    preprocessor = Molmo2InputsPreprocessor(chat_mode=True)
    inputs = {"token_type_ids": torch.tensor([token_type_ids])}
    assert preprocessor.cache_prefix_length(inputs, prefix_len) == expected
