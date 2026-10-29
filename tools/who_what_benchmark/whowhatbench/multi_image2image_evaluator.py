# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import datasets
from huggingface_hub import snapshot_download

from .im2im_evaluator import Image2ImageEvaluator
from .registry import register_evaluator


@register_evaluator("multi-image-to-image")
class MultiImage2ImageEvaluator(Image2ImageEvaluator):
    def _prepare_default_data(self):
        dataset_path = snapshot_download(
            "OmniGen2/OmniContext",
            repo_type="dataset",
            revision="0f6aabbe8d3448d51184093372e980205d9980d3",
            allow_patterns=["data-*.arrow"],
        )
        default_dataset = (
            datasets.load_dataset(
                "arrow",
                data_files=os.path.join(dataset_path, "data-*.arrow"),
                split="train",
                streaming=True,
            )
            .filter(lambda example: len(example["input_images"]) >= 2)
            .take(10 if self.num_samples is None else self.num_samples)
        )
        return default_dataset.map(
            lambda example: {
                "prompts": example["instruction"],
                "images": example["input_images"],
            },
            remove_columns=default_dataset.column_names,
        )

    def _resolve_conditioning_images(self, images):
        if not isinstance(images, (list, tuple)):
            images = [images]
        if len(images) < 2:
            raise ValueError(f"multi-image-to-image requires at least 2 conditioning images, got {len(images)}")
        resolve_image = super()._resolve_conditioning_images
        return [resolve_image(image) for image in images]
