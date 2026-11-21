# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import torch
from openvino_genai import ClassificationActivation
from optimum.intel import OVModelForVisualSequenceClassification
from PIL import Image
from transformers import AutoProcessor, PretrainedConfig


def apply_activation(logits: np.ndarray, activation: ClassificationActivation) -> np.ndarray:
    if activation == ClassificationActivation.SOFTMAX:
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)
    if activation == ClassificationActivation.SIGMOID:
        return 1.0 / (1.0 + np.exp(-logits))
    return logits


def default_activation(config: PretrainedConfig) -> ClassificationActivation:
    """Mirrors transformers' TextClassificationPipeline default."""
    if config.problem_type == "regression":
        return ClassificationActivation.NONE
    if config.problem_type == "multi_label_classification" or config.num_labels == 1:
        return ClassificationActivation.SIGMOID
    return ClassificationActivation.SOFTMAX


class OptimumClassifier:
    """Reference classifier on optimum-intel for a model exported with `--task text-classification`."""

    def __init__(self, model_path: Path):
        self.model = OVModelForVisualSequenceClassification.from_pretrained(model_path, device="CPU")
        self.processor = AutoProcessor.from_pretrained(model_path)

    @property
    def config(self) -> PretrainedConfig:
        return self.model.config

    def format_prompt(self, prompt: str, num_images: int) -> str:
        """Renders the chat template with the model's native image tags, without a generation prompt."""
        content = [{"type": "image"}] * num_images + [{"type": "text", "text": prompt}]
        return self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False)

    def classify(self, text: str, images: list[np.ndarray]) -> np.ndarray:
        """Returns the logits [1, num_labels] of an already-formatted prompt, adding no special tokens."""
        inputs = self.processor(
            text=[text],
            images=[Image.fromarray(image) for image in images] or None,
            add_special_tokens=False,
            return_tensors="pt",
        )
        with torch.no_grad():
            return self.model(**inputs).logits.numpy()
