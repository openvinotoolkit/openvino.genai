# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datasets

from .im2im_evaluator import Image2ImageEvaluator
from .registry import register_evaluator


@register_evaluator("multi-image-to-image")
class MultiImage2ImageEvaluator(Image2ImageEvaluator):
    DEFAULT_NUM_INFERENCE_STEPS = 10
    DEFAULT_IMAGE_SIZE = 512
    DEFAULT_GENERATION_SETTINGS = [
        {"width": 512, "height": 512, "guidance_scale": 1.0, "negative_prompt": None},
        {"width": 512, "height": 512, "guidance_scale": 2.0, "negative_prompt": ""},
        {"width": 512, "height": 512, "guidance_scale": 1.0, "negative_prompt": None},
        {"width": 512, "height": 512, "guidance_scale": 1.0, "negative_prompt": None},
        {"width": 512, "height": 512, "guidance_scale": 1.0, "negative_prompt": None},
        {"width": 512, "height": 256, "guidance_scale": 4.0, "negative_prompt": "blurry, low quality, distorted"},
        {"width": 512, "height": 512, "guidance_scale": 1.0, "negative_prompt": None},
    ]

    def _prepare_default_data(self):
        default_dataset = datasets.load_dataset(
            "OmniGen2/OmniContext",
            revision="0f6aabbe8d3448d51184093372e980205d9980d3",
            split="train",
            streaming=True,
        )
        default_dataset = default_dataset.filter(lambda example: 2 <= len(example["input_images"]) <= 3).take(
            7 if self.num_samples is None else self.num_samples
        )
        return default_dataset.map(
            lambda example, index: {
                "prompts": example["instruction"],
                "images": example["input_images"],
                **self.DEFAULT_GENERATION_SETTINGS[index % len(self.DEFAULT_GENERATION_SETTINGS)],
            },
            with_indices=True,
            remove_columns=default_dataset.column_names,
        )

    def _resolve_conditioning_images(self, images):
        if not isinstance(images, (list, tuple)):
            images = [images]
        if len(images) < 2:
            raise ValueError(f"multi-image-to-image requires at least 2 conditioning images, got {len(images)}")
        resolve_image = super()._resolve_conditioning_images
        return [resolve_image(image) for image in images]
