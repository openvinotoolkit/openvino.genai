# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Molmo2: the prompt layout (the exported checkpoint chat template places the media, as Molmo2Processor does), the
Molmo2Processor options read from processor_config.json, chat with cached images, and the native genai-cli exporter
(numerical parity with the Hugging Face decoder and the export options)."""

import json
import os
import shutil
import subprocess  # nosec B404
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import openvino
import pytest
from huggingface_hub import snapshot_download
from openvino_genai import ChatHistory, GenerationConfig, Tokenizer, VLMPipeline
from openvino_genai._cli import uv_env
from openvino_genai._cli.exporters.molmo2 import Molmo2Exporter

from test_vlm_pipeline import (  # noqa: F401 (fixtures)
    MODEL_MOLMO2,
    _get_ov_model,
    car_tensor,
    cat_image,
    cat_tensor,
    handwritten_tensor,
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_chat_recomputes_cached_images_when_an_image_is_added(
    molmo2_dir: Path,
    backend: str,
    cat_tensor: openvino.Tensor,
    car_tensor: openvino.Tensor,
    handwritten_tensor: openvino.Tensor,
):
    # The third image goes right after the first two, which stay in the cached prefix. Image tokens attend to each
    # other bidirectionally, so the cached states of the first two images must be computed again.
    # The greedy texts of the tiny model do not depend on the image states: the second turn compares beam search scores
    # (log probabilities) with the full history instead. f32 everywhere: the CPU plugin compresses the KV cache by
    # default, which alone changes the scores of cached tokens.
    properties = {"ATTENTION_BACKEND": backend, "KV_CACHE_PRECISION": "f32", "INFERENCE_PRECISION_HINT": "f32"}
    beam_search = GenerationConfig(max_new_tokens=4, num_beams=2, num_return_sequences=1)
    pipe = VLMPipeline(molmo2_dir, "CPU", **properties)
    history = ChatHistory()
    history.append({"role": "user", "content": "Compare."})
    first = pipe.generate(history, images=[cat_tensor, car_tensor], generation_config=_config()).texts[0]
    history.append({"role": "assistant", "content": first})
    history.append({"role": "user", "content": "And now?"})
    second = pipe.generate(history, images=[handwritten_tensor], generation_config=beam_search)

    expected_layout = f"Image 1<|image|>Image 2<|image|>Image 3<|image|>User: Compare. {first}User: And now? Assistant:"
    beam_search.apply_chat_template = False
    reference = VLMPipeline(molmo2_dir, "CPU", **properties).generate(
        expected_layout, images=[cat_tensor, car_tensor, handwritten_tensor], generation_config=beam_search
    )
    assert second.texts == reference.texts
    assert second.scores == pytest.approx(reference.scores, abs=1e-4)


def _config_dir(molmo2_dir: Path, config_dir: Path, config_file: str, options: dict) -> Path:
    """A copy of the configuration files of the model with ``options`` set in ``config_file``."""
    config_dir.mkdir()
    for file in molmo2_dir.iterdir():
        if file.is_file() and file.suffix not in (".xml", ".bin"):
            shutil.copy(file, config_dir)
    config = json.loads((config_dir / config_file).read_text(encoding="utf-8"))
    config.update(options)
    (config_dir / config_file).write_text(json.dumps(config), encoding="utf-8")
    return config_dir


@pytest.mark.parametrize(
    "config_file, options, message",
    [
        ("preprocessor_config.json", {"overlap_margins": [2]}, "overlap_margins"),
        ("preprocessor_config.json", {"pooling_size": [2, -1]}, "pooling_size"),
        ("video_preprocessor_config.json", {"pooling_size": "3x3"}, "pooling_size"),
        ("video_preprocessor_config.json", {"sampling_fps": -2}, "sampling_fps"),
        ("video_preprocessor_config.json", {"num_frames": 0}, "num_frames"),
    ],
)
def test_malformed_preprocessing_config_is_rejected(
    molmo2_dir: Path, molmo2_models: dict, tmp_path: Path, config_file: str, options: dict, message: str
):
    config_dir = _config_dir(molmo2_dir, tmp_path / "config", config_file, options)
    with pytest.raises(RuntimeError, match=message):
        VLMPipeline(molmo2_models, Tokenizer(molmo2_dir), config_dir, "CPU")


@pytest.mark.parametrize("backend", BACKENDS)
def test_patch_token_mismatch_with_vision_embeddings_is_rejected(
    molmo2_dir: Path, molmo2_models: dict, tmp_path: Path, backend: str, cat_tensor: openvino.Tensor
):
    # A config whose image_patch_id is not the "<im_patch>" token the processor emits: the text has fewer patch tokens
    # than the vision encoder produced embeddings. Like the reference implementation, this fails instead of dropping
    # the surplus embeddings.
    config = json.loads((molmo2_dir / "config.json").read_text(encoding="utf-8"))
    config_dir = _config_dir(molmo2_dir, tmp_path / "config", "config.json", {"image_patch_id": config["image_col_id"]})
    pipe = VLMPipeline(molmo2_models, Tokenizer(molmo2_dir), config_dir, "CPU", ATTENTION_BACKEND=backend)
    with pytest.raises(RuntimeError, match=r"the text has \d+ image patch tokens, but the vision encoders produced"):
        pipe.generate("Describe.", images=[cat_tensor], generation_config=_config())


@pytest.mark.parametrize(
    "media, option, value, more_tokens",
    [
        ("images", "image_use_col_tokens", False, False),  # no column token after each row of patches
        # The low-resolution crop has them with image_use_col_tokens off.
        ("images", "use_single_crop_col_tokens", True, True),
        ("videos", "video_use_col_tokens", True, True),  # a column token after each row of patches of every frame
    ],
)
def test_processor_config_options_change_visual_tokens(
    molmo2_dir: Path,
    molmo2_models: dict,
    tmp_path: Path,
    cat_tensor: openvino.Tensor,
    synthetic_video_32x32_tensor: openvino.Tensor,
    media: str,
    option: str,
    value: bool,
    more_tokens: bool,
):
    # Molmo2Processor saves these options in processor_config.json, not in the image or video processor configs.
    tokenizer = Tokenizer(molmo2_dir)
    inputs = {"images": [cat_tensor]} if media == "images" else {"videos": [synthetic_video_32x32_tensor]}

    def num_input_tokens(processor_options: dict) -> int:
        name = "-".join(f"{key}-{val}" for key, val in processor_options.items()) or "defaults"
        config_dir = _config_dir(molmo2_dir, tmp_path / name, "processor_config.json", processor_options)
        pipe = VLMPipeline(molmo2_models, tokenizer, config_dir, "CPU")
        result = pipe.generate("Describe.", **inputs, generation_config=_config())
        return result.perf_metrics.get_num_input_tokens()

    # use_single_crop_col_tokens only matters when it differs from image_use_col_tokens.
    base = {"image_use_col_tokens": False} if option == "use_single_crop_col_tokens" else {}
    baseline = num_input_tokens(base)
    changed = num_input_tokens({**base, option: value})
    assert changed != baseline
    assert (changed > baseline) == more_tokens


def test_exported_language_model_matches_hugging_face(molmo2_dir: Path):
    """genai-cli rewrites the decoder blocks before torch.export (_patch_text_model_for_export) and builds the image
    attention mask in the graph: the exported language model must give the logits of the original Hugging Face decoder,
    for a prefill with text and image tokens and for cached decoding."""
    checkpoint = retry_request(lambda: snapshot_download(MODEL_MOLMO2))
    command = ["python", "-m", "utils.molmo2_hf_parity", checkpoint, str(molmo2_dir)]
    requirement = f"transformers=={Molmo2Exporter.TRANSFORMERS_VERSION}"
    env = dict(os.environ)
    if uv_env.is_satisfied(requirement):
        command[0] = sys.executable
    else:
        # The Molmo2 remote code needs the transformers version of the exporter: provide it like genai-cli does.
        uv_command = [uv_env.find_uv(), "run", "--no-project", "--with", requirement]
        if sys.prefix != sys.base_prefix:
            env["VIRTUAL_ENV"] = sys.prefix
            uv_command.append("--active")
        else:
            uv_command += ["--python", sys.executable]
        command = uv_command + command
    subprocess.run(command, cwd=Path(__file__).parent, env=env, check=True)


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
