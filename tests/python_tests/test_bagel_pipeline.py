# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Tests for BagelPipeline: visual question answering, text-to-image generation and image editing.

A tiny random BAGEL model is generated locally and exported with optimum-cli. The GenAI outputs are compared with
optimum-intel's OVModelForVisualCausalLM, which is tested against transformers in optimum-intel. The native genai-cli
export of the same checkpoint is checked against the optimum-cli one.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import openvino as ov
import pytest
import torch
from optimum.intel import OVModelForVisualCausalLM
from PIL import Image
from transformers import AutoProcessor

import openvino_genai as ov_genai
from utils.bagel_test_assets import (
    prepare_tiny_bagel_genai_cli_path,
    prepare_tiny_bagel_ov_frames_path,
    prepare_tiny_bagel_ov_path,
)
from utils.constants import get_ov_cache_converted_models_dir

PROMPT = "What is shown in this image?"
IMAGE_PROMPT = "A red bicycle leaning against a blue wall"
EDIT_PROMPT = "Make it green"
IMAGE_SIZE = 32
NUM_STEPS = 4

pytestmark = pytest.mark.bagel


@pytest.fixture(scope="module")
def bagel_ov_path() -> Path:
    return prepare_tiny_bagel_ov_path(get_ov_cache_converted_models_dir())


@pytest.fixture(scope="module")
def genai_pipe(bagel_ov_path: Path) -> ov_genai.BagelPipeline:
    # no tokenizer_path: the tokenizer is read from the export directory
    return ov_genai.BagelPipeline(bagel_ov_path, "CPU", INFERENCE_PRECISION_HINT="f32")


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


@pytest.fixture(scope="module")
def images() -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    # the last image is smaller, so the edited image (the VAE size of the last input) differs from the others
    return [rng.integers(0, 256, shape, dtype=np.uint8) for shape in ((48, 64, 3), (64, 48, 3), (20, 20, 3))]


def to_uint8(image: torch.Tensor) -> np.ndarray:
    """[1, 3, H, W] in [-1, 1] -> [H, W, 3] u8, as BagelPipeline converts the VAE output."""
    pixels = ((image[0].permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0.0, 1.0)
    return (pixels * 255.0).astype(np.uint8)


def optimum_text(optimum_model, processor, inputs, max_new_tokens: int) -> str:
    output = optimum_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    tokens = output[0, inputs["input_ids"].shape[1] :].tolist()
    eos = optimum_model.generation_config.eos_token_id
    if eos in tokens:
        tokens = tokens[: tokens.index(eos)]
    return processor.tokenizer.decode(tokens, skip_special_tokens=False)


def test_text_matches_optimum(genai_pipe, optimum_model, processor):
    max_new_tokens = 8
    genai_text = genai_pipe.generate_text(PROMPT, max_new_tokens=max_new_tokens)

    inputs = optimum_model.preprocess_inputs(text=PROMPT, processor=processor)
    assert genai_text == optimum_text(optimum_model, processor, inputs, max_new_tokens)


def test_multi_image_text_matches_optimum(genai_pipe, optimum_model, processor, images):
    max_new_tokens = 8
    genai_text = genai_pipe.generate_text(PROMPT, [ov.Tensor(i[None]) for i in images], max_new_tokens=max_new_tokens)

    content = [{"type": "image"}] * len(images) + [{"type": "text", "text": PROMPT}]
    prompt = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
    inputs = processor(text=prompt, images=[Image.fromarray(i) for i in images], return_tensors="pt")
    assert genai_text == optimum_text(optimum_model, processor, inputs, max_new_tokens)


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


def test_edit_image_matches_optimum(genai_pipe, optimum_model, processor, image):
    seed = 3
    inputs = processor(text=EDIT_PROMPT, images=[Image.fromarray(image)], generation_mode="image", return_tensors="pt")
    height, width = inputs["vae_pixel_values"].shape[-2:]
    num_tokens = (height // optimum_model.latent_downsample) * (width // optimum_model.latent_downsample)
    latents = torch.from_numpy(
        np.random.RandomState(seed).randn(num_tokens, optimum_model._patch_dim).astype(np.float32)
    )

    edits = {}
    for cfg_img_scale in (1.0, 2.0):
        config = ov_genai.BagelImageGenerationConfig()
        config.num_inference_steps = NUM_STEPS
        config.cfg_img_scale = cfg_img_scale
        config.seed = seed
        genai_image = genai_pipe.edit_image(ov.Tensor(image[None]), EDIT_PROMPT, config).data[0]

        optimum_image = optimum_model.generate(
            **inputs,
            generation_mode="image",
            num_inference_steps=NUM_STEPS,
            guidance_scale=config.cfg_text_scale,
            image_guidance_scale=cfg_img_scale,
            cfg_interval=(0.0, 1.0),
            latents=latents,
        )

        assert genai_image.shape == (height, width, 3)
        diff = np.abs(genai_image.astype(np.int32) - to_uint8(optimum_image).astype(np.int32))
        assert diff.max() <= 1, f"cfg_img_scale={cfg_img_scale}: max abs pixel diff {diff.max()}"
        edits[cfg_img_scale] = genai_image
    assert not np.array_equal(edits[1.0], edits[2.0]), "cfg_img_scale has no effect"


def test_multi_image_edit_matches_reference(genai_pipe, optimum_model, processor, images, monkeypatch):
    """optimum-intel edits one image. The reference keeps its guidance loop and adds, as the interleaved inference of
    the original BAGEL, a VAE and a ViT block per image to the contexts that hold the input image."""
    seed = 5
    config = ov_genai.BagelImageGenerationConfig()
    config.num_inference_steps = NUM_STEPS
    config.seed = seed
    genai_image = genai_pipe.edit_image([ov.Tensor(i[None]) for i in images], EDIT_PROMPT, config).data[0]

    per_image = [
        processor(text=EDIT_PROMPT, images=[Image.fromarray(i)], generation_mode="image", return_tensors="pt")
        for i in images
    ]
    add_vae_image, add_vit_image = optimum_model._add_vae_image, optimum_model._add_vit_image

    def add_images(context, _):
        for inputs in per_image:
            add_vae_image(context, inputs["vae_pixel_values"])
            add_vit_image(context, inputs["pixel_values"], inputs["image_position_ids"])

    monkeypatch.setattr(optimum_model, "_add_vae_image", add_images)
    monkeypatch.setattr(optimum_model, "_add_vit_image", lambda *args: None)
    last = per_image[-1]
    height, width = last["vae_pixel_values"].shape[-2:]
    num_tokens = (height // optimum_model.latent_downsample) * (width // optimum_model.latent_downsample)
    latents = torch.from_numpy(
        np.random.RandomState(seed).randn(num_tokens, optimum_model._patch_dim).astype(np.float32)
    )
    optimum_image = optimum_model.generate(
        **last,
        generation_mode="image",
        num_inference_steps=NUM_STEPS,
        guidance_scale=config.cfg_text_scale,
        image_guidance_scale=config.cfg_img_scale,
        cfg_interval=(0.0, 1.0),
        latents=latents,
    )

    assert (height, width) != tuple(per_image[0]["vae_pixel_values"].shape[-2:])
    assert genai_image.shape == (height, width, 3)
    diff = np.abs(genai_image.astype(np.int32) - to_uint8(optimum_image).astype(np.int32))
    assert diff.max() <= 1, f"max abs pixel diff {diff.max()}"


def reference_frames(optimum_model, processor, frames, prompt, num_frames, config, monkeypatch) -> list[np.ndarray]:
    """optimum-intel's guidance loop over a window of the input frames' count: an input frame is a VAE and a ViT block,
    a generated frame its clean latent tokens only. The reference decodes the generated frames one by one."""
    per_frame = [
        processor(text=prompt, images=[Image.fromarray(f)], generation_mode="image", return_tensors="pt")
        for f in frames
    ]
    window = []
    for inputs in per_frame:
        latent = optimum_model.vae_encoder({"pixel_values": inputs["vae_pixel_values"].numpy()})["latent"]
        height, width = inputs["vae_pixel_values"].shape[-2:]
        window.append((optimum_model._patchify(latent), (height, width), inputs))
    height, width = window[-1][1]

    def add_window(context, _):
        for tokens, (h, w), inputs in window:
            optimum_model._latent_block(context, tokens, 0.0, optimum_model._latent_position_ids(h, w), True)
            if inputs is not None:
                add_vit_image(context, inputs["pixel_values"], inputs["image_position_ids"])

    add_vit_image = optimum_model._add_vit_image
    monkeypatch.setattr(optimum_model, "_add_vae_image", add_window)
    monkeypatch.setattr(optimum_model, "_add_vit_image", lambda *args: None)
    num_tokens = (height // optimum_model.latent_downsample) * (width // optimum_model.latent_downsample)
    generated = []
    for k in range(num_frames):
        noise = np.random.RandomState(config.seed + k).randn(num_tokens, optimum_model._patch_dim)
        tokens = optimum_model.generate(
            **per_frame[-1],
            generation_mode="image",
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.cfg_text_scale,
            image_guidance_scale=config.cfg_img_scale,
            cfg_interval=(0.0, 1.0),
            latents=torch.from_numpy(noise.astype(np.float32)),
            output_type="latent",
        ).numpy()
        generated.append(tokens)
        window = window[1:] + [(tokens, (height, width), None)]
    monkeypatch.undo()

    return [
        to_uint8(
            torch.from_numpy(
                optimum_model.vae_decoder({"latents": optimum_model._unpatchify(tokens, height, width)})["image"]
            )
        )
        for tokens in generated
    ]


def test_generate_frames_matches_reference(genai_pipe, optimum_model, processor, images, monkeypatch):
    num_frames = 3
    config = ov_genai.BagelImageGenerationConfig()
    config.num_inference_steps = NUM_STEPS
    config.seed = 11
    genai_pipe.reset_perf_metrics()
    genai_frames = genai_pipe.generate_frames([ov.Tensor(i[None]) for i in images], EDIT_PROMPT, num_frames, config)
    decoder_calls = [stage for stage, _ in genai_pipe.get_perf_metrics().infer_ms if stage == "vae_dec"]
    assert len(decoder_calls) == 1, "the generated frames are not decoded in one call"

    expected = reference_frames(optimum_model, processor, images, EDIT_PROMPT, num_frames, config, monkeypatch)
    assert genai_frames.get_element_type() == ov.Type.u8
    assert list(genai_frames.get_shape()) == [num_frames, *expected[0].shape]
    for k, (genai_frame, frame) in enumerate(zip(genai_frames.data, expected)):
        diff = np.abs(genai_frame.astype(np.int32) - frame.astype(np.int32))
        assert diff.max() <= 1, f"frame {k}: max abs pixel diff {diff.max()}"
    assert not np.array_equal(genai_frames.data[0], genai_frames.data[1])


def test_generate_frames_decodes_per_frame_without_dynamic_batch(genai_pipe, bagel_ov_path, images, tmp_path):
    """Exports before the dynamic VAE decoder batch decode one frame per call, with the same frames."""
    decoder = ov.Core().read_model(bagel_ov_path / "openvino_vae_decoder_model.xml")
    shape = decoder.input().get_partial_shape()
    assert shape[0].is_dynamic
    shape[0] = 1
    decoder.reshape({decoder.input(): shape})
    for file in bagel_ov_path.iterdir():
        if not file.name.startswith("openvino_vae_decoder_model."):
            (tmp_path / file.name).symlink_to(file)
    ov.save_model(decoder, tmp_path / "openvino_vae_decoder_model.xml", compress_to_fp16=False)
    static_pipe = ov_genai.BagelPipeline(tmp_path, "CPU", INFERENCE_PRECISION_HINT="f32")

    config = ov_genai.BagelImageGenerationConfig()
    config.num_inference_steps = NUM_STEPS
    frames = [ov.Tensor(i[None]) for i in images]
    per_frame = static_pipe.generate_frames(frames, EDIT_PROMPT, 2, config).data
    batched = genai_pipe.generate_frames(frames, EDIT_PROMPT, 2, config).data
    assert per_frame.shape == batched.shape
    diff = np.abs(per_frame.astype(np.int32) - batched.astype(np.int32))
    assert diff.max() <= 1, f"max abs pixel diff {diff.max()}"
    decoder_calls = [stage for stage, _ in static_pipe.get_perf_metrics().infer_ms if stage == "vae_dec"]
    assert len(decoder_calls) == 2


def test_generate_frames_with_frames_models(genai_pipe, images):
    """An export with the 3-frame models encodes the 3 same-size input frames and decodes the 3 generated frames in one
    call each, with the frames of the single-frame models."""
    frames_pipe = ov_genai.BagelPipeline(
        prepare_tiny_bagel_ov_frames_path(get_ov_cache_converted_models_dir()), "CPU", INFERENCE_PRECISION_HINT="f32"
    )
    frames = [ov.Tensor(np.ascontiguousarray(i[None, :20, :20])) for i in images]
    config = ov_genai.BagelImageGenerationConfig()
    config.num_inference_steps = NUM_STEPS
    config.seed = 5

    frames_pipe.reset_perf_metrics()
    batched = frames_pipe.generate_frames(frames, EDIT_PROMPT, 3, config).data
    perf = frames_pipe.get_perf_metrics()
    stages = [stage for stage, _ in perf.infer_ms]
    assert [stages.count(stage) for stage in ("vit", "vae_enc", "vae_dec")] == [1, 1, 1]
    compiled = {name for name, _ in perf.compile_ms}
    assert {f"compile_model {name}_3img" for name in ("vision_embeddings", "vae_encoder", "vae_decoder")} <= compiled

    single = genai_pipe.generate_frames(frames, EDIT_PROMPT, 3, config).data
    assert batched.shape == single.shape
    diff = np.abs(batched.astype(np.int32) - single.astype(np.int32))
    assert diff.max() <= 1, f"max abs pixel diff {diff.max()}"

    # 2 generated frames have no 2-frame decoder: one call of the dynamic batch decoder
    frames_pipe.reset_perf_metrics()
    frames_pipe.generate_frames(frames, EDIT_PROMPT, 2, config)
    assert [stage for stage, _ in frames_pipe.get_perf_metrics().infer_ms].count("vae_dec") == 1


def test_generate_frames_labels_match_optimum(genai_pipe, optimum_model, processor, images):
    """Frame labels as the original OpenVINO BAGEL world model demo, with text and image guidance."""
    frames = [np.ascontiguousarray(i[:20, :20]) for i in images]
    labels = ["Frame 1 (t-2):", "Frame 2 (t-1):", "Frame 3 (t, current):"]
    config = ov_genai.BagelImageGenerationConfig()
    config.num_inference_steps = NUM_STEPS
    config.seed = 3
    config.frame_labels = labels
    genai_frames = genai_pipe.generate_frames([ov.Tensor(f[None]) for f in frames], EDIT_PROMPT, 2, config).data

    inputs = processor(
        text=EDIT_PROMPT, images=[Image.fromarray(f) for f in frames], generation_mode="image", return_tensors="pt"
    )
    label_ids = [processor(text=label, generation_mode="image", return_tensors="pt")["input_ids"] for label in labels]
    optimum_frames = optimum_model.generate_frames(
        **inputs,
        num_frames=2,
        num_inference_steps=NUM_STEPS,
        guidance_scale=config.cfg_text_scale,
        image_guidance_scale=config.cfg_img_scale,
        seed=config.seed,
        frame_labels=label_ids,
    )
    for k, (genai_frame, frame) in enumerate(zip(genai_frames, optimum_frames)):
        diff = np.abs(genai_frame.astype(np.int32) - to_uint8(frame[None]).astype(np.int32))
        assert diff.max() <= 1, f"frame {k}: max abs pixel diff {diff.max()}"

    config.frame_labels = []
    unlabeled = genai_pipe.generate_frames([ov.Tensor(f[None]) for f in frames], EDIT_PROMPT, 2, config).data
    assert not np.array_equal(unlabeled, genai_frames)
    config.frame_labels = labels[:2]
    with pytest.raises(RuntimeError, match="one label per input frame"):
        genai_pipe.generate_frames([ov.Tensor(f[None]) for f in frames], EDIT_PROMPT, 2, config)


def test_generate_frames_rejects_invalid_arguments(genai_pipe, image):
    with pytest.raises(RuntimeError, match="at least one input frame"):
        genai_pipe.generate_frames([], EDIT_PROMPT, 1)
    with pytest.raises(RuntimeError, match="num_frames must be positive"):
        genai_pipe.generate_frames([ov.Tensor(image[None])], EDIT_PROMPT, 0)


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


def test_strided_image_matches_packed(genai_pipe, image):
    height, width = image.shape[:2]
    wide = np.zeros((1, height, width + 16, 3), np.uint8)
    wide[0, :, 8 : 8 + width] = image
    roi = ov.Tensor(ov.Tensor(wide), ov.Coordinate([0, 0, 8, 0]), ov.Coordinate([1, height, 8 + width, 3]))
    assert not roi.is_continuous()

    config = ov_genai.BagelImageGenerationConfig()
    config.num_inference_steps = NUM_STEPS
    packed = genai_pipe.edit_image(ov.Tensor(image[None]), EDIT_PROMPT, config).data
    np.testing.assert_array_equal(genai_pipe.edit_image(roi, EDIT_PROMPT, config).data, packed)


def test_perf_metrics_during_generation(genai_pipe):
    config = ov_genai.BagelImageGenerationConfig()
    config.size = IMAGE_SIZE
    config.num_inference_steps = NUM_STEPS
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(genai_pipe.generate_image, IMAGE_PROMPT, config)
        while not future.done():
            genai_pipe.get_perf_metrics()
            genai_pipe.reset_perf_metrics()
        future.result()
    genai_pipe.reset_perf_metrics()
    genai_pipe.generate_image(IMAGE_PROMPT, config)
    assert {"vae_dec"} <= {stage for stage, _ in genai_pipe.get_perf_metrics().infer_ms}


def test_genai_cli_export_matches_optimum_export(genai_pipe, image):
    cli_pipe = ov_genai.BagelPipeline(
        prepare_tiny_bagel_genai_cli_path(get_ov_cache_converted_models_dir()), "CPU", INFERENCE_PRECISION_HINT="f32"
    )
    tensor = ov.Tensor(image[None])
    assert cli_pipe.generate_text(PROMPT, tensor, max_new_tokens=8) == genai_pipe.generate_text(
        PROMPT, tensor, max_new_tokens=8
    )

    config = ov_genai.BagelImageGenerationConfig()
    config.size = IMAGE_SIZE
    config.num_inference_steps = NUM_STEPS
    for run in (
        lambda pipe: pipe.generate_image(IMAGE_PROMPT, config),
        lambda pipe: pipe.edit_image(tensor, EDIT_PROMPT, config),
    ):
        diff = np.abs(run(cli_pipe).data.astype(np.int32) - run(genai_pipe).data.astype(np.int32))
        assert diff.max() <= 1, f"max abs pixel diff {diff.max()}"


def test_rejects_other_models(tmp_path: Path, bagel_ov_path: Path):
    config = json.loads((bagel_ov_path / "config.json").read_text())
    config["model_type"] = "qwen2_vl"
    for file in bagel_ov_path.iterdir():
        (tmp_path / file.name).symlink_to(file)
    (tmp_path / "config.json").unlink()
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="BagelPipeline expects a BAGEL model"):
        ov_genai.BagelPipeline(tmp_path, "CPU")
