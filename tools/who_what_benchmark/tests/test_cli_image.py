# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import sys
import pytest
import logging
import tempfile
import re
from types import SimpleNamespace

import numpy as np

from conftest import convert_model, run_wwb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CACHE = tempfile.mkdtemp()
OV_IMAGE_MODELS = [
    "optimum-intel-internal-testing/tiny-random-stable-diffusion-xl",
    "optimum-intel-internal-testing/stable-diffusion-3-tiny-random",
    "optimum-intel-internal-testing/tiny-random-flux",
    "optimum-intel-internal-testing/tiny-random-flux-fill",
]


def get_similarity(output: str) -> float:
    METRIC_PATTERN = "INFO:whowhatbench.wwb:   similarity"
    substr = output[output.find(METRIC_PATTERN) + len(METRIC_PATTERN) + 1:]
    float_pattern = r"[-+]?\d*\.\d+"
    matches = re.findall(float_pattern, substr)
    return float(matches[-1])


def test_text2image_genai_npu_reshapes_before_compile(monkeypatch):
    from whowhatbench import model_loaders

    calls = []

    class FakePipeline:
        def __init__(self, model_dir):
            calls.append(("init", model_dir))

        def get_generation_config(self):
            calls.append(("get_generation_config",))
            return SimpleNamespace(guidance_scale=7.5)

        def reshape(self, **kwargs):
            calls.append(("reshape", kwargs))

        def compile(self, device, **properties):
            calls.append(("compile", device, properties))

    class FakeGenAIModelWrapper:
        def __init__(self, model, model_dir, model_type):
            self.model = model
            self.model_dir = model_dir
            self.model_type = model_type

    monkeypatch.setitem(
        sys.modules,
        "openvino_genai",
        SimpleNamespace(Text2ImagePipeline=FakePipeline),
    )
    monkeypatch.setattr(model_loaders, "GenAIModelWrapper", FakeGenAIModelWrapper)

    wrapper = model_loaders.load_text2image_genai_pipeline(
        "model_dir",
        device="NPU",
        ov_config={"CACHE_DIR": "cache"},
        image_size=512,
    )

    assert wrapper.model_type == "text-to-image"
    assert calls == [
        ("init", "model_dir"),
        ("get_generation_config",),
        (
            "reshape",
            {"num_images_per_prompt": 1, "height": 512, "width": 512, "guidance_scale": 7.5},
        ),
        ("compile", "NPU", {"CACHE_DIR": "cache"}),
    ]


def test_text2image_genai_npu_requires_positive_image_size(monkeypatch):
    from whowhatbench import model_loaders

    monkeypatch.setitem(sys.modules, "openvino_genai", SimpleNamespace(Text2ImagePipeline=object))

    with pytest.raises(ValueError, match="positive --image-size"):
        model_loaders.load_text2image_genai_pipeline("model_dir", device="NPU", ov_config={})


def test_genai_gen_image_uses_model_adapter_config():
    from whowhatbench.wwb import genai_gen_image

    calls = []

    class FakeModel:
        resolution = (512, 512)
        adapter_config = object()

        def generate(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return SimpleNamespace(data=np.zeros((1, 1, 1, 3), dtype=np.uint8))

    genai_gen_image(FakeModel(), "prompt", 4)

    assert calls[0][1]["adapters"] is FakeModel.adapter_config


@pytest.mark.parametrize(
    ("model_id", "model_type", "backend"),
    [
        ("hf-internal-testing/tiny-stable-diffusion-torch", "image-to-image", "hf"),
        ("hf-internal-testing/tiny-stable-diffusion-xl-pipe", "image-to-image", "hf"),
        ("hf-internal-testing/tiny-stable-diffusion-torch", "text-to-image", "hf"),
        ("hf-internal-testing/tiny-stable-diffusion-torch", "text-to-image", "openvino"),
        ("hf-internal-testing/tiny-stable-diffusion-xl-pipe", "text-to-image", "hf"),
        ("hf-internal-testing/tiny-stable-diffusion-torch", "image-inpainting", "hf"),
        ("hf-internal-testing/tiny-stable-diffusion-xl-pipe", "image-inpainting", "hf"),
    ],
)
def test_image_model_types(model_id, model_type, backend, tmp_path):
    if 'tiny-stable-diffusion-torch' in model_id and sys.platform == 'darwin':
        pytest.xfail("Ticket 173169")
    if (model_type == "image-to-image" or model_type == "image-inpainting") and sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    wwb_args = [
        "--base-model",
        model_id,
        "--target-model",
        model_id,
        "--num-samples",
        "1",
        "--gt-data",
        tmp_path / "test_sd.csv",
        "--device",
        "CPU",
        "--model-type",
        model_type,
        "--num-inference-steps",
        "2",
    ]
    if backend == "hf":
        wwb_args.append("--hf")
    elif backend == "genai":
        wwb_args.append("--genai")

    output = run_wwb(wwb_args)

    assert "Metrics for model" in output
    similarity = get_similarity(output)
    assert similarity >= 0.98


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    list(itertools.product(OV_IMAGE_MODELS,
                           ["image-to-image",
                            "text-to-image",
                            "image-inpainting"
                            ])),
)
def test_image_model_genai(model_id, model_type, tmp_path):
    if ("flux-fill" in model_id) and (model_type != "image-inpainting"):
        pytest.skip(reason="FLUX-Fill is supported as inpainting only")
    if model_id == "optimum-intel-internal-testing/tiny-random-flux" and model_type == "image-to-image":
        pytest.xfail("Randomly wwb died with <Signals.SIGABRT: 6>. Ticket 170878")
    if (model_type == "image-to-image" or model_type == "image-inpainting") and sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    mac_arm64_skip = any(substring in model_id for substring in ('stable-diffusion-xl',
                                                                 'tiny-random-stable-diffusion',
                                                                 'stable-diffusion-3',
                                                                 'tiny-random-flux'))

    if mac_arm64_skip and sys.platform == 'darwin':
        pytest.xfail("Ticket 173169")

    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = convert_model(model_id)

    run_wwb([
        "--base-model",
        model_id,
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
        "--num-inference-steps",
        "2",
    ])
    assert GT_FILE.exists()
    assert (tmp_path / "reference").exists()

    output = run_wwb(
        [
            "--target-model",
            MODEL_PATH,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
            "--num-inference-steps",
            "2",
            "--taylorseer-config",
            '{"disable_cache_after_step": 0}',
            "--output",
            tmp_path,
        ]
    )

    assert "Metrics for model" in output
    similarity = get_similarity(output)
    assert similarity >= 0.97751  # Ticket 166496
    assert (tmp_path / "target").exists()

    run_wwb(
        [
            "--target-model",
            MODEL_PATH,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--output",
            tmp_path,
            "--genai",
            "--num-inference-steps",
            "2",
            "--taylorseer-config",
            '{"disable_cache_after_step": 0}',
        ]
    )
    assert (tmp_path / "target").exists()
    assert (tmp_path / "target.csv").exists()

    # test w/o models
    run_wwb([
        "--target-data",
        tmp_path / "target.csv",
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
        "--num-inference-steps",
        "2",
    ])


def test_image_model_genai_with_taylorseer(tmp_path):
    if sys.platform == "darwin":
        pytest.xfail("Ticket 173169")

    model_id = "optimum-intel-internal-testing/stable-diffusion-3-tiny-random"
    model_type = "text-to-image"

    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = convert_model(model_id)

    run_wwb(
        [
            "--base-model",
            model_id,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--num-inference-steps",
            "4",
        ]
    )
    assert GT_FILE.exists()
    assert (tmp_path / "reference").exists()

    # Test with TaylorSeer partial config
    output = run_wwb(
        [
            "--target-model",
            MODEL_PATH,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
            "--num-inference-steps",
            "4",
            "--taylorseer-config",
            '{"cache_interval": 5, "disable_cache_before_step": 2, "disable_cache_after_step": 4}',
        ]
    )

    assert "Metrics for model" in output
    assert "TaylorSeer config:" in output
    similarity = get_similarity(output)
    assert similarity >= 0.98


@pytest.mark.parametrize(
    ("model_id", "model_type", "backend"),
    [
        ("hf-internal-testing/tiny-stable-diffusion-torch", "text-to-image", "hf"),
    ],
)
def test_image_custom_dataset(model_id, model_type, backend, tmp_path):
    if sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    GT_FILE = tmp_path / "test_sd.csv"
    wwb_args = [
        "--base-model",
        model_id,
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
        "--dataset",
        "google-research-datasets/conceptual_captions",
        "--dataset-field",
        "caption",
        "--num-inference-steps",
        "2",
    ]
    if backend == "hf":
        wwb_args.append("--hf")
    elif backend == "genai":
        wwb_args.append("--genai")

    run_wwb(wwb_args)

    assert os.path.exists(GT_FILE)
