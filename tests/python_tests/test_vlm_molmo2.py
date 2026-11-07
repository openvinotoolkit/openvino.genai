# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Molmo2 prompt layout: the exported checkpoint chat template places the media, as Molmo2Processor does, and the
Molmo2Processor options are read from processor_config.json."""

import json
import shutil
from pathlib import Path

import numpy as np
import openvino
import pytest
from openvino_genai import ChatHistory, GenerationConfig, Tokenizer, VLMPipeline

from test_vlm_pipeline import MODEL_MOLMO2, _get_ov_model, car_tensor, cat_image, cat_tensor  # noqa: F401 (fixtures)

BACKENDS = ["SDPA", "PA"]


@pytest.fixture(scope="module")
def molmo2_dir() -> Path:
    return Path(_get_ov_model(MODEL_MOLMO2))


@pytest.fixture(scope="module")
def molmo2_models(molmo2_dir: Path) -> dict[str, tuple[str, openvino.Tensor]]:
    """The IRs of the model in the ModelsMap form, to build pipelines with other config files."""
    models = {}
    for xml in molmo2_dir.glob("openvino_*_model.xml"):
        name = xml.stem.removeprefix("openvino_").removesuffix("_model")
        models[name] = (
            xml.read_text(encoding="utf-8"),
            openvino.Tensor(np.fromfile(xml.with_suffix(".bin"), np.uint8)),
        )
    return models


def _config(apply_chat_template: bool = True) -> GenerationConfig:
    return GenerationConfig(max_new_tokens=8, do_sample=False, apply_chat_template=apply_chat_template)


def test_checkpoint_chat_template_is_kept(molmo2_dir: Path):
    tokenizer = VLMPipeline(molmo2_dir, "CPU").get_tokenizer()
    exported_template = (molmo2_dir / "chat_template.jinja").read_text(encoding="utf-8")
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Bye"},
    ]
    rendered = tokenizer.apply_chat_template(history, add_generation_prompt=True)
    assert rendered == tokenizer.apply_chat_template(
        history, add_generation_prompt=True, chat_template=exported_template
    )
    assert rendered == "User: Hi HelloUser: Bye Assistant:"


@pytest.mark.parametrize("backend", BACKENDS)
def test_chat_template_places_images_like_molmo2_processor(
    molmo2_dir: Path, backend: str, cat_tensor: openvino.Tensor, car_tensor: openvino.Tensor
):
    # For several images the checkpoint template emits "Image N<|image|>" in front of the "User:" turn.
    pipe = VLMPipeline(molmo2_dir, "CPU", ATTENTION_BACKEND=backend)
    images = [cat_tensor, car_tensor]
    templated = pipe.generate("Compare the images.", images=images, generation_config=_config())
    expected_layout = "Image 1<|image|>Image 2<|image|>User: Compare the images. Assistant:"
    raw = pipe.generate(expected_layout, images=images, generation_config=_config(apply_chat_template=False))
    assert templated.perf_metrics.get_num_input_tokens() == raw.perf_metrics.get_num_input_tokens()
    assert templated.texts == raw.texts


def test_system_message_is_merged_into_the_user_turn(molmo2_dir: Path, cat_tensor: openvino.Tensor):
    # The template has no system role: the system message goes in front of the first user message.
    history = ChatHistory()
    history.append({"role": "system", "content": "Be brief."})
    history.append({"role": "user", "content": "Describe."})
    chat = VLMPipeline(molmo2_dir, "CPU").generate(history, images=[cat_tensor], generation_config=_config())
    expected_layout = "<|image|>User: Be brief.\nDescribe. Assistant:"
    raw_pipe = VLMPipeline(molmo2_dir, "CPU")
    raw = raw_pipe.generate(expected_layout, images=[cat_tensor], generation_config=_config(apply_chat_template=False))
    assert chat.texts == raw.texts


@pytest.mark.parametrize("backend", BACKENDS)
def test_chat_reembeds_images_the_template_moves(
    molmo2_dir: Path, backend: str, cat_tensor: openvino.Tensor, car_tensor: openvino.Tensor
):
    # An image of the second turn goes in front of the first "User:" turn together with the image of the first turn,
    # so the first image is embedded again.
    pipe = VLMPipeline(molmo2_dir, "CPU", ATTENTION_BACKEND=backend)
    history = ChatHistory()
    history.append({"role": "user", "content": "Describe."})
    first = pipe.generate(history, images=[cat_tensor], generation_config=_config()).texts[0]
    history.append({"role": "assistant", "content": first})
    history.append({"role": "user", "content": "And this one?"})
    second = pipe.generate(history, images=[car_tensor], generation_config=_config()).texts

    expected_layout = f"Image 1<|image|>Image 2<|image|>User: Describe. {first}User: And this one? Assistant:"
    reference_pipe = VLMPipeline(molmo2_dir, "CPU", ATTENTION_BACKEND=backend)
    raw_config = _config(apply_chat_template=False)
    reference = reference_pipe.generate(expected_layout, images=[cat_tensor, car_tensor], generation_config=raw_config)
    assert second == reference.texts


@pytest.mark.parametrize(
    "option, value, more_tokens",
    [
        ("image_use_col_tokens", False, False),  # no column token after each row of patches
        ("use_single_crop_col_tokens", True, True),  # the low-resolution crop has them with image_use_col_tokens off
    ],
)
def test_processor_config_options_change_image_tokens(
    molmo2_dir: Path,
    molmo2_models: dict,
    tmp_path: Path,
    cat_tensor: openvino.Tensor,
    option: str,
    value: bool,
    more_tokens: bool,
):
    tokenizer = Tokenizer(molmo2_dir)

    def num_input_tokens(processor_options: dict) -> int:
        config_dir = tmp_path / ("-".join(f"{key}-{val}" for key, val in processor_options.items()) or "defaults")
        config_dir.mkdir()
        for file in molmo2_dir.iterdir():
            if file.is_file() and file.suffix not in (".xml", ".bin"):
                shutil.copy(file, config_dir)
        processor = json.loads((config_dir / "processor_config.json").read_text(encoding="utf-8"))
        processor.update(processor_options)
        (config_dir / "processor_config.json").write_text(json.dumps(processor), encoding="utf-8")
        pipe = VLMPipeline(molmo2_models, tokenizer, config_dir, "CPU")
        result = pipe.generate("Describe.", images=[cat_tensor], generation_config=_config())
        return result.perf_metrics.get_num_input_tokens()

    # use_single_crop_col_tokens only matters when it differs from image_use_col_tokens.
    base = {"image_use_col_tokens": False} if option == "use_single_crop_col_tokens" else {}
    baseline = num_input_tokens(base)
    changed = num_input_tokens({**base, option: value})
    assert changed != baseline
    assert (changed > baseline) == more_tokens
