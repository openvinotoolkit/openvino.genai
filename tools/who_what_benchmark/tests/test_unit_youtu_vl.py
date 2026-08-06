# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from PIL import Image


def test_youtu_vl_preprocessor_registered():
    """youtu_vl must resolve to a VLM inputs preprocessor so the HF visual-text
    path does not fall through to the optimum-intel mapping (which would raise
    KeyError('youtu_vl'))."""
    from whowhatbench.inputs_preprocessors import (
        MODEL_TYPE_TO_CLS_MAPPING,
        YoutuVLInputsPreprocessor,
        VLMInputsPreprocessor,
    )

    assert "youtu_vl" in MODEL_TYPE_TO_CLS_MAPPING
    cls = MODEL_TYPE_TO_CLS_MAPPING["youtu_vl"]
    assert cls is YoutuVLInputsPreprocessor
    assert issubclass(cls, VLMInputsPreprocessor)


def test_youtu_vl_preprocessor_builds_multimodal_inputs():
    """The youtu_vl preprocessor should render the chat template and forward the
    image to the processor (Qwen2-VL style), without any Hub-id special casing."""
    from whowhatbench.inputs_preprocessors import YoutuVLInputsPreprocessor

    captured = {}

    class FakeProcessor:
        def apply_chat_template(self, conversation, add_generation_prompt, tokenize):
            captured["conversation"] = conversation
            return "rendered-prompt"

        def __call__(self, images, text, videos, return_tensors):
            captured["images"] = images
            captured["text"] = text
            captured["videos"] = videos
            return {"input_ids": [[1, 2, 3]]}

    img = Image.new("RGB", (8, 8), color=(10, 20, 30))
    pre = YoutuVLInputsPreprocessor()
    out = pre.preprocess_inputs("What is this?", image=img, processor=FakeProcessor())

    assert out == {"input_ids": [[1, 2, 3]]}
    assert captured["text"] == "rendered-prompt"
    assert captured["images"] is img
    assert captured["videos"] is None
    # user message must carry both the image and the text content.
    content_types = [c["type"] for c in captured["conversation"][0]["content"]]
    assert "image" in content_types and "text" in content_types


def test_load_prompts_local_csv_visual_text(tmp_path):
    """--dataset pointing at a local CSV with prompts/images/videos should be
    loaded generically, resolving image paths relative to the CSV directory."""
    from whowhatbench import wwb

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "a.png"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(img_path)

    csv_path = tmp_path / "inputs.csv"
    csv_path.write_text(
        "prompts,images,videos\n"
        "Describe this,images/a.png,\n"
    )

    args = SimpleNamespace(dataset=str(csv_path), model_type="visual-text",
                           split=None, dataset_field="prompts")
    res = wwb.load_prompts(args)

    assert res["prompts"] == ["Describe this"]
    assert isinstance(res["images"][0], Image.Image)
    assert res["images"][0].mode == "RGB"
    assert res["videos"] == [None]


def test_load_prompts_local_csv_requires_prompts_column(tmp_path):
    from whowhatbench import wwb

    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("questions,images\nhi,x.png\n")
    args = SimpleNamespace(dataset=str(csv_path), model_type="visual-text",
                           split=None, dataset_field="prompts")
    try:
        wwb.load_prompts(args)
    except ValueError as e:
        assert "prompts" in str(e)
    else:
        raise AssertionError("expected ValueError for missing 'prompts' column")
