# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from whowhatbench import wwb
from whowhatbench.image2video_evaluator import Image2VideoEvaluator


def _make_dummy_image(path, color=(12, 34, 56)):
    img = Image.new("RGB", (8, 8), color=color)
    img.save(path)


def _patch_video_runtime(monkeypatch):
    monkeypatch.setattr("whowhatbench.text2video_evaluator.VideoSimilarity", lambda: None)
    monkeypatch.setattr(
        "whowhatbench.image2video_evaluator.export_to_video",
        lambda frames, video_path, fps: None,
    )


def test_create_evaluator_image2video_forwards_csv_dataset(monkeypatch, tmp_path):
    csv_path = tmp_path / "i2v.csv"
    pd.DataFrame({"prompt": ["a cat walking"], "images": ["0.png"]}).to_csv(csv_path, index=False)

    captured = {}

    class DummyEvaluator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(wwb.EVALUATOR_REGISTRY, "image-to-video", DummyEvaluator)

    args = SimpleNamespace(
        model_type="image-to-video",
        dataset=str(csv_path),
        dataset_field="text",
        split=None,
        gt_data=str(tmp_path / "gt.csv"),
        num_samples=1,
        num_inference_steps=2,
        video_frames_num=5,
        genai=False,
        seed=42,
        empty_adapters=False,
        image_dir=str(tmp_path),
    )

    wwb.create_evaluator(base_model="dummy_model", args=args)

    assert captured["test_data"] == str(csv_path)
    assert captured["image_dir"] == str(tmp_path)


def test_load_prompts_image2video_hf_dataset_preserves_video_columns(monkeypatch):
    class DummyHFDataset:
        def __init__(self):
            self.column_names = ["text", "negative_prompt", "width", "height", "guidance_scale", "image"]
            self._cols = {
                "text": ["p1", "p2"],
                "negative_prompt": ["neg1", "neg2"],
                "width": [512, 704],
                "height": [320, 480],
                "guidance_scale": [3.0, 4.0],
                "image": ["a.png", "b.png"],
            }

        def __getitem__(self, key):
            return self._cols[key]

    monkeypatch.setattr(wwb, "load_dataset", lambda path, name, split: DummyHFDataset())

    args = SimpleNamespace(
        model_type="image-to-video",
        dataset="dummy/hf_i2v_dataset",
        dataset_field="text",
        split="test",
    )

    prompts = wwb.load_prompts(args)
    assert prompts == {
        "prompt": ["p1", "p2"],
        "negative_prompt": ["neg1", "neg2"],
        "width": [512, 704],
        "height": [320, 480],
        "guidance_scale": [3.0, 4.0],
        "image": ["a.png", "b.png"],
    }


def test_image2video_evaluator_csv_resolves_images_and_fallback_to_image_dir(monkeypatch, tmp_path):
    _patch_video_runtime(monkeypatch)

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    _make_dummy_image(img_dir / "custom.png", color=(200, 10, 20))
    _make_dummy_image(img_dir / "1.png", color=(10, 200, 20))

    csv_path = tmp_path / "i2v.csv"
    pd.DataFrame(
        {
            "prompt": ["first prompt", "second prompt"],
            "images": ["custom.png", ""],
            "width": [320, 320],
            "height": [240, 240],
        }
    ).to_csv(csv_path, index=False)

    calls = []

    def fake_gen_video_fn(model, prompt, image, **kwargs):
        calls.append((prompt, np.array(image)[0, 0].tolist(), kwargs["width"], kwargs["height"]))
        return [image]

    evaluator = Image2VideoEvaluator(
        base_model=object(),
        gt_data=str(tmp_path / "gt.csv"),
        test_data=str(csv_path),
        gen_video_fn=fake_gen_video_fn,
        image_dir=str(img_dir),
    )

    assert evaluator.gt_data["prompt"].tolist() == ["first prompt", "second prompt"]
    assert calls == [
        ("first prompt", [200, 10, 20], 320, 240),
        ("second prompt", [10, 200, 20], 320, 240),
    ]


def test_image2video_evaluator_csv_resolves_relative_to_csv_dir_without_image_dir(monkeypatch, tmp_path):
    _patch_video_runtime(monkeypatch)

    sub_dir = tmp_path / "dataset_dir"
    sub_dir.mkdir()
    _make_dummy_image(sub_dir / "rel.png", color=(50, 60, 70))

    csv_path = sub_dir / "i2v.csv"
    pd.DataFrame({"prompts": ["prompt from prompts col"], "image": ["rel.png"]}).to_csv(csv_path, index=False)

    seen_colors = []

    def fake_gen_video_fn(model, prompt, image, **kwargs):
        seen_colors.append((prompt, np.array(image)[0, 0].tolist()))
        return [image]

    evaluator = Image2VideoEvaluator(
        base_model=object(),
        gt_data=str(tmp_path / "gt.csv"),
        test_data=str(csv_path),
        gen_video_fn=fake_gen_video_fn,
    )

    assert evaluator.gt_data["prompt"].tolist() == ["prompt from prompts col"]
    assert seen_colors == [("prompt from prompts col", [50, 60, 70])]


def test_image2video_evaluator_image_dir_without_dataset_skips_remote_download(monkeypatch, tmp_path):
    _patch_video_runtime(monkeypatch)

    def fail_prepare_default_data(*args, **kwargs):
        raise AssertionError("prepare_default_data should not be called when --image-dir is provided")

    monkeypatch.setattr("whowhatbench.image2video_evaluator.prepare_default_data", fail_prepare_default_data)

    _make_dummy_image(tmp_path / "0.png", color=(90, 80, 70))

    calls = []

    def fake_gen_video_fn(model, prompt, image, **kwargs):
        calls.append((prompt, np.array(image)[0, 0].tolist()))
        return [image]

    evaluator = Image2VideoEvaluator(
        base_model=object(),
        gt_data=str(tmp_path / "gt.csv"),
        test_data=None,
        num_samples=1,
        gen_video_fn=fake_gen_video_fn,
        image_dir=str(tmp_path),
    )

    assert len(calls) == 1
    assert calls[0][1] == [90, 80, 70]
    assert len(evaluator.gt_data) == 1


def test_image2video_evaluator_rejects_missing_prompt_column(monkeypatch, tmp_path):
    _patch_video_runtime(monkeypatch)

    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"wrong_col": ["hello"]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="must contain a 'prompt' or 'prompts' column"):
        Image2VideoEvaluator(
            base_model=object(),
            gt_data=str(tmp_path / "gt.csv"),
            test_data=str(csv_path),
            image_dir=str(tmp_path),
        )
