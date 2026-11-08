# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Molmo2: the prompt layout (the exported checkpoint chat template places the media, as Molmo2Processor does), the
Molmo2Processor options read from processor_config.json, and the export options of the native genai-cli exporter."""

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import openvino
import pytest
from huggingface_hub import snapshot_download
from openvino_genai import ChatHistory, GenerationConfig, Tokenizer, VLMPipeline

from test_vlm_pipeline import (  # noqa: F401 (fixtures)
    MODEL_MOLMO2,
    _get_ov_model,
    car_tensor,
    cat_image,
    cat_tensor,
    synthetic_video,
    synthetic_video_32x32,
    synthetic_video_32x32_tensor,
)
from utils.export import export_with_genai_cli
from utils.network import retry_request

BACKENDS = ["SDPA", "PA"]


@pytest.fixture(scope="module")
def molmo2_dir() -> Path:
    return Path(_get_ov_model(MODEL_MOLMO2))


@pytest.fixture(scope="module")
def molmo2_pipe(molmo2_dir: Path) -> Callable[[str], VLMPipeline]:
    """Pipelines shared by the tests that generate from string prompts only.

    On SDPA, generate(ChatHistory) followed by generate(str) on one VLMPipeline fails for every VLM, so the chat tests
    run the chat on a pipeline of their own.
    """
    pipelines = {}

    def get(backend: str) -> VLMPipeline:
        if backend not in pipelines:
            pipelines[backend] = VLMPipeline(molmo2_dir, "CPU", ATTENTION_BACKEND=backend)
        return pipelines[backend]

    return get


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
    tokenizer = Tokenizer(molmo2_dir)
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
    molmo2_pipe: Callable[[str], VLMPipeline],
    backend: str,
    cat_tensor: openvino.Tensor,
    car_tensor: openvino.Tensor,
):
    # For several images the checkpoint template emits "Image N<|image|>" in front of the "User:" turn.
    pipe = molmo2_pipe(backend)
    images = [cat_tensor, car_tensor]
    templated = pipe.generate("Compare the images.", images=images, generation_config=_config())
    expected_layout = "Image 1<|image|>Image 2<|image|>User: Compare the images. Assistant:"
    raw = pipe.generate(expected_layout, images=images, generation_config=_config(apply_chat_template=False))
    assert templated.perf_metrics.get_num_input_tokens() == raw.perf_metrics.get_num_input_tokens()
    assert templated.texts == raw.texts


@pytest.mark.parametrize("backend", BACKENDS)
def test_chat_template_places_video_like_molmo2_processor(
    molmo2_pipe: Callable[[str], VLMPipeline], backend: str, synthetic_video_32x32_tensor: openvino.Tensor
):
    pipe = molmo2_pipe(backend)
    videos = [synthetic_video_32x32_tensor]
    templated = pipe.generate("Describe the video.", videos=videos, generation_config=_config())
    expected_layout = "<|video|>User: Describe the video. Assistant:"
    raw = pipe.generate(expected_layout, videos=videos, generation_config=_config(apply_chat_template=False))
    assert templated.perf_metrics.get_num_input_tokens() == raw.perf_metrics.get_num_input_tokens()
    assert templated.texts == raw.texts


def test_system_message_is_merged_into_the_user_turn(
    molmo2_dir: Path, molmo2_pipe: Callable[[str], VLMPipeline], cat_tensor: openvino.Tensor
):
    # The template has no system role: the system message goes in front of the first user message.
    history = ChatHistory()
    history.append({"role": "system", "content": "Be brief."})
    history.append({"role": "user", "content": "Describe."})
    chat = VLMPipeline(molmo2_dir, "CPU", ATTENTION_BACKEND="PA").generate(
        history, images=[cat_tensor], generation_config=_config()
    )
    expected_layout = "<|image|>User: Be brief.\nDescribe. Assistant:"
    raw = molmo2_pipe("PA").generate(
        expected_layout, images=[cat_tensor], generation_config=_config(apply_chat_template=False)
    )
    assert chat.texts == raw.texts


@pytest.mark.parametrize("backend", BACKENDS)
def test_chat_reembeds_images_the_template_moves(
    molmo2_dir: Path,
    molmo2_pipe: Callable[[str], VLMPipeline],
    backend: str,
    cat_tensor: openvino.Tensor,
    car_tensor: openvino.Tensor,
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
    reference = molmo2_pipe(backend).generate(
        expected_layout, images=[cat_tensor, car_tensor], generation_config=_config(apply_chat_template=False)
    )
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


def _weight_types(xml: Path) -> set[str]:
    model = openvino.Core().read_model(xml)
    return {op.get_element_type().get_type_name() for op in model.get_ops() if op.get_type_name() == "Constant"}


def test_native_export_keeps_float_weights_by_default(molmo2_dir: Path):
    # The tiny model has less than 1B weights, so without --weight-format no submodel is compressed.
    for name in ("language", "text_embeddings", "vision_embeddings"):
        assert not _weight_types(molmo2_dir / f"openvino_{name}_model.xml") & {"u4", "i4", "u8", "i8"}


@pytest.mark.parametrize(
    "options, language_weights, tokenizer_converted",
    [
        # int4_asym for the language model; the other submodels get int8_sym.
        (["--weight-format", "int4", "--disable-convert-tokenizer"], "u4", False),
        (["--weight-format", "int8"], "u8", True),
        (["--weight-format", "int8", "--sym"], "i8", True),
    ],
)
def test_native_export_options(tmp_path: Path, options: list[str], language_weights: str, tokenizer_converted: bool):
    checkpoint = retry_request(lambda: snapshot_download(MODEL_MOLMO2))
    output = tmp_path / "ov"
    export_with_genai_cli(checkpoint, output, ["--trust-remote-code", *options])

    assert language_weights in _weight_types(output / "openvino_language_model.xml")
    for name in ("text_embeddings", "vision_embeddings"):
        assert "i8" in _weight_types(output / f"openvino_{name}_model.xml")
    assert (output / "openvino_tokenizer.xml").is_file() == tokenizer_converted
    assert (output / "openvino_detokenizer.xml").is_file() == tokenizer_converted
