# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Tests for BagelPipeline: visual question answering, text-to-image generation and image editing.

A tiny random BAGEL model is generated locally and exported with optimum-cli. The GenAI outputs are compared with
optimum-intel's OVModelForVisualCausalLM, which is tested against transformers in optimum-intel.
"""

import json
from pathlib import Path

import numpy as np
import openvino as ov
import pytest
import torch
from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoProcessor

import openvino_genai as ov_genai
from utils.bagel_test_assets import prepare_tiny_bagel_ov_path
from utils.constants import get_ov_cache_converted_models_dir

PROMPT = "What is shown in this image?"
IMAGE_PROMPT = "A red bicycle leaning against a blue wall"
IMAGE_SIZE = 32
NUM_STEPS = 4

pytestmark = pytest.mark.bagel


@pytest.fixture(scope="module")
def bagel_ov_path() -> Path:
    return prepare_tiny_bagel_ov_path(get_ov_cache_converted_models_dir())


@pytest.fixture(scope="module")
def genai_pipe(bagel_ov_path: Path) -> ov_genai.BagelPipeline:
    return ov_genai.BagelPipeline(bagel_ov_path, "CPU", bagel_ov_path, INFERENCE_PRECISION_HINT="f32")


@pytest.fixture(scope="module")
def optimum_model(bagel_ov_path: Path) -> OVModelForVisualCausalLM:
    return OVModelForVisualCausalLM.from_pretrained(bagel_ov_path, ov_config={"INFERENCE_PRECISION_HINT": "f32"})


@pytest.fixture(scope="module")
def processor(bagel_ov_path: Path):
    return AutoProcessor.from_pretrained(bagel_ov_path)


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)


def to_uint8(image: torch.Tensor) -> np.ndarray:
    """[1, 3, H, W] in [-1, 1] -> [H, W, 3] u8, as BagelPipeline converts the VAE output."""
    pixels = ((image[0].permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0.0, 1.0)
    return (pixels * 255.0).astype(np.uint8)


def test_text_matches_optimum(genai_pipe, optimum_model, processor):
    max_new_tokens = 8
    genai_text = genai_pipe.generate_text(PROMPT, max_new_tokens=max_new_tokens)

    inputs = optimum_model.preprocess_inputs(text=PROMPT, processor=processor)
    output = optimum_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    tokens = output[0, inputs["input_ids"].shape[1] :].tolist()
    eos = optimum_model.generation_config.eos_token_id
    if eos in tokens:
        tokens = tokens[: tokens.index(eos)]
    assert genai_text == processor.tokenizer.decode(tokens, skip_special_tokens=False)


def test_text_to_image_matches_optimum(genai_pipe, optimum_model, processor):
    seed = 7
    config = ov_genai.BagelImageGenerationConfig()
    config.size = IMAGE_SIZE
    config.num_inference_steps = NUM_STEPS
    config.seed = seed
    genai_image = genai_pipe.generate_image(IMAGE_PROMPT, config).data[0]

    # BagelPipeline draws its initial noise as numpy.random.RandomState(seed).randn
    num_tokens = (IMAGE_SIZE // optimum_model.latent_downsample) ** 2
    latents = torch.from_numpy(
        np.random.RandomState(seed).randn(num_tokens, optimum_model._patch_dim).astype(np.float32)
    )
    inputs = processor(text=IMAGE_PROMPT, generation_mode="image", return_tensors="pt")
    optimum_image = optimum_model.generate(
        **inputs,
        generation_mode="image",
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        num_inference_steps=NUM_STEPS,
        latents=latents,
    )

    assert genai_image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    diff = np.abs(genai_image.astype(np.int32) - to_uint8(optimum_image).astype(np.int32))
    assert diff.max() <= 1, f"max abs pixel diff {diff.max()}"


def test_image_inputs(genai_pipe, image):
    tensor = ov.Tensor(image[None])
    assert isinstance(genai_pipe.generate_text(PROMPT, tensor, max_new_tokens=4), str)

    config = ov_genai.BagelImageGenerationConfig()
    config.num_inference_steps = NUM_STEPS
    edited = genai_pipe.edit_image(tensor, "Make it green", config)
    # the output keeps the VAE resolution of the input: a 64x48 image resizes to 32x32 (vae_max_size=32, stride 16)
    assert edited.get_element_type() == ov.Type.u8
    assert list(edited.get_shape()) == [1, 32, 32, 3]

    metrics = genai_pipe.get_perf_metrics()
    stages = {stage for stage, _ in metrics.infer_ms}
    assert {"vit", "vae_enc", "vae_dec"} <= stages
    genai_pipe.reset_perf_metrics()
    assert not genai_pipe.get_perf_metrics().infer_ms


def test_rejects_other_models(tmp_path: Path, bagel_ov_path: Path):
    config = json.loads((bagel_ov_path / "config.json").read_text())
    config["model_type"] = "qwen2_vl"
    for file in bagel_ov_path.iterdir():
        (tmp_path / file.name).symlink_to(file)
    (tmp_path / "config.json").unlink()
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="BagelPipeline expects a BAGEL model"):
        ov_genai.BagelPipeline(tmp_path, "CPU")
