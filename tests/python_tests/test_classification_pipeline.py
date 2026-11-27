# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

import numpy as np
import openvino as ov
import pytest
from openvino_genai import ClassificationActivation, ClassificationPipeline
from huggingface_hub import snapshot_download

from utils.atomic_download import AtomicDownloadManager
from utils.classification import OptimumClassifier, apply_activation, default_activation
from utils.constants import get_ov_cache_converted_models_dir
from utils.hugging_face import export_with_optimum_cli, sanitize_model_id
from utils.network import retry_request

CLASSIFICATION_MODELS = [
    "optimum-intel-internal-testing/tiny-random-minicpm-ocr-jev",
]

PROMPTS = [
    "Classify the input.",
    "Is the text in the image correct? Answer yes or no.",
]


def random_image(seed: int) -> np.ndarray:
    # A size that needs no resize: GenAI and the HF processor interpolate differently.
    return np.random.default_rng(seed).integers(0, 256, (256, 256, 3), dtype=np.uint8)


@pytest.fixture(scope="module", params=CLASSIFICATION_MODELS)
def model_path(request: pytest.FixtureRequest) -> Path:
    model_id = request.param
    path = get_ov_cache_converted_models_dir() / f"{sanitize_model_id(model_id)}_text-classification"
    AtomicDownloadManager(path).execute(
        lambda temp_path: export_with_optimum_cli(
            retry_request(lambda: snapshot_download(model_id)),
            "text-classification",
            temp_path,
            trust_remote_code=False,
        )
    )
    return path


@pytest.fixture(scope="module")
def pipeline(model_path: Path) -> ClassificationPipeline:
    return ClassificationPipeline(model_path, "CPU")


@pytest.fixture(scope="module")
def optimum_classifier(model_path: Path) -> OptimumClassifier:
    return OptimumClassifier(model_path)


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("num_images", [0, 1, 2], ids=["text", "one_image", "two_images"])
def test_classify_matches_optimum(
    pipeline: ClassificationPipeline, optimum_classifier: OptimumClassifier, prompt: str, num_images: int
):
    images = [random_image(seed) for seed in range(num_images)]
    # classify() applies no chat template, so both pipelines get the same prompt with the model's native image tags.
    text = optimum_classifier.format_prompt(prompt, num_images)

    reference_logits = optimum_classifier.classify(text, images)
    result = pipeline.classify(text, [ov.Tensor(image) for image in images])

    logits = result.logits.data
    # macOS ARM CPU inference can use reduced precision for image processing.
    atol = 5e-4 if sys.platform == "darwin" and num_images else 1e-5
    np.testing.assert_allclose(logits, reference_logits, atol=atol)
    np.testing.assert_allclose(
        result.scores, apply_activation(logits, default_activation(optimum_classifier.config)), rtol=1e-5
    )


def test_result_is_unchanged_after_repeated_classification(pipeline: ClassificationPipeline):
    first_result = pipeline.classify(PROMPTS[0])
    first_logits = first_result.logits.data.copy()
    first_scores = np.asarray(first_result.scores).copy()

    pipeline.classify(PROMPTS[1], [ov.Tensor(random_image(seed=0))])

    np.testing.assert_array_equal(first_result.logits.data, first_logits)
    np.testing.assert_array_equal(first_result.scores, first_scores)


@pytest.mark.parametrize(
    "activation",
    [ClassificationActivation.NONE, ClassificationActivation.SIGMOID, ClassificationActivation.SOFTMAX],
    ids=lambda activation: activation.name,
)
@pytest.mark.parametrize("use_config", [False, True], ids=["kwargs", "config"])
def test_activation(
    model_path: Path, pipeline: ClassificationPipeline, activation: ClassificationActivation, use_config: bool
):
    if use_config:
        pipe = ClassificationPipeline(
            model_path, "CPU", ClassificationPipeline.Config(classification_activation=activation)
        )
    else:
        pipe = ClassificationPipeline(model_path, "CPU", classification_activation=activation)

    result = pipe.classify(PROMPTS[0])

    reference_logits = pipeline.classify(PROMPTS[0]).logits.data
    np.testing.assert_array_equal(result.logits.data, reference_logits)
    np.testing.assert_allclose(result.scores, apply_activation(reference_logits, activation), rtol=1e-5)
