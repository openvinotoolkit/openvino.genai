# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from PIL import Image

from whowhatbench.image2video_evaluator import Image2VideoEvaluator
from whowhatbench.text2video_evaluator import Text2VideoEvaluator
from whowhatbench.wwb import load_prompts


def _configure_evaluator(evaluator, test_data):
    evaluator.test_data = test_data
    evaluator.num_samples = None
    evaluator.num_inference_steps = 1
    evaluator.num_frames = 9
    evaluator.frame_rate = 25
    evaluator.seed = 42
    evaluator.decode_timestep = 0.4
    evaluator.decode_noise_scale = 0.7
    evaluator.is_genai = False
    evaluator.empty_adapters = False


def _common_test_data():
    return {
        "prompt": ["dataset values"],
        "negative_prompt": [""],
        "width": [32],
        "height": [32],
        "guidance_scale": [1.0],
        "decode_timestep": [0.0],
        "decode_noise_scale": [1.0],
    }


def test_text2video_decode_parameters_use_dataset_values_with_cli_fallback(monkeypatch, tmp_path):
    evaluator = Text2VideoEvaluator.__new__(Text2VideoEvaluator)
    _configure_evaluator(evaluator, _common_test_data())
    calls = []

    def generate_video(_model, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr("whowhatbench.text2video_evaluator.export_to_video", lambda *_args: None)
    evaluator._generate_data(None, generate_video, str(tmp_path))

    assert [(call["decode_timestep"], call["decode_noise_scale"]) for call in calls] == [
        (0.0, 1.0),
    ]


def test_text2video_missing_optional_columns_use_defaults(monkeypatch, tmp_path):
    evaluator = Text2VideoEvaluator.__new__(Text2VideoEvaluator)
    _configure_evaluator(
        evaluator,
        {
            "prompt": ["minimal dataset"],
            "decode_timestep": [0.2],
            "decode_noise_scale": [0.3],
        },
    )
    calls = []

    def generate_video(_model, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr("whowhatbench.text2video_evaluator.export_to_video", lambda *_args: None)
    result = evaluator._generate_data(None, generate_video, str(tmp_path))

    assert result[["negative_prompt", "width", "height", "guidance_scale"]].to_dict("records") == [
        {
            "negative_prompt": "",
            "width": Text2VideoEvaluator.DEF_WIDTH,
            "height": Text2VideoEvaluator.DEF_HEIGHT,
            "guidance_scale": Text2VideoEvaluator.DEF_GUIDANCE_SCALE,
        }
    ]
    actual_generation_values = [
        (call["negative_prompt"], call["width"], call["height"], call["guidance_scale"]) for call in calls
    ]
    assert actual_generation_values == [
        (
            "",
            Text2VideoEvaluator.DEF_WIDTH,
            Text2VideoEvaluator.DEF_HEIGHT,
            Text2VideoEvaluator.DEF_GUIDANCE_SCALE,
        )
    ]


def test_image2video_decode_parameters_fall_back_when_dataset_columns_are_missing(monkeypatch, tmp_path):
    evaluator = Image2VideoEvaluator.__new__(Image2VideoEvaluator)
    test_data = _common_test_data()
    del test_data["decode_timestep"]
    del test_data["decode_noise_scale"]
    test_data["images"] = [Image.new("RGB", (32, 32))]
    _configure_evaluator(evaluator, test_data)
    evaluator.image_dir = None
    calls = []

    def generate_video(_model, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr("whowhatbench.image2video_evaluator.export_to_video", lambda *_args: None)
    evaluator._generate_data(None, generate_video, str(tmp_path))

    assert [(call["decode_timestep"], call["decode_noise_scale"]) for call in calls] == [
        (0.4, 0.7),
    ]


def test_load_prompts_preserves_video_dataset_columns(monkeypatch):
    class Dataset:
        @staticmethod
        def to_dict():
            return {
                "caption": ["prompt"],
                "decode_timestep": [0.2],
                "decode_noise_scale": [0.3],
            }

    class Args:
        dataset = "video-dataset"
        split = "test"
        dataset_field = "caption"
        model_type = "text-to-video"

    monkeypatch.setattr("whowhatbench.wwb.load_dataset", lambda **_kwargs: Dataset())

    assert load_prompts(Args()) == {
        "prompt": ["prompt"],
        "decode_timestep": [0.2],
        "decode_noise_scale": [0.3],
    }
