import types
from unittest import mock

import pytest
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from whowhatbench import wwb


def _args(model_id):
    return types.SimpleNamespace(base_model=None, target_model=model_id)


class _FakeTokenizer(PreTrainedTokenizerBase):
    """Minimal stand-in that satisfies isinstance(..., PreTrainedTokenizerBase)."""

    def __init__(self):
        pass


class _FakeImageProcessor:
    def __call__(self, images=None, return_tensors=None):
        return {"pixel_values": [0]}


def test_load_processor_falls_back_to_image_processor_for_vision_text():
    """GLM-Edge-V ships only a text tokenizer via AutoProcessor. For vision-text
    configs, load_processor must return an image processor so the OV target path
    (optimum-intel preprocess_inputs) can call processor(images=...)."""
    config = types.SimpleNamespace(model_type="glm", vision_config=object())
    image_processor = _FakeImageProcessor()

    with mock.patch.object(wwb.AutoConfig, "from_pretrained", return_value=config), \
         mock.patch.object(wwb.AutoProcessor, "from_pretrained", return_value=_FakeTokenizer()), \
         mock.patch.object(wwb.AutoImageProcessor, "from_pretrained", return_value=image_processor):
        preprocessor, out_config = wwb.load_processor(_args("dummy-glm-edge-v"))

    assert preprocessor is image_processor
    assert not isinstance(preprocessor, PreTrainedTokenizerBase)
    assert out_config is config


def test_load_processor_keeps_real_processor_for_unified_vlm():
    """When AutoProcessor returns a real (non-tokenizer) multimodal processor,
    load_processor must keep it and not override with an image processor."""
    config = types.SimpleNamespace(model_type="llava", vision_config=object())
    real_processor = _FakeImageProcessor()  # not a PreTrainedTokenizerBase

    with mock.patch.object(wwb.AutoConfig, "from_pretrained", return_value=config), \
         mock.patch.object(wwb.AutoProcessor, "from_pretrained", return_value=real_processor), \
         mock.patch.object(wwb.AutoImageProcessor, "from_pretrained") as img_proc:
        preprocessor, _ = wwb.load_processor(_args("dummy-llava"))

    assert preprocessor is real_processor
    img_proc.assert_not_called()


def test_load_processor_ignores_fallback_for_text_only_model():
    """A text-only model (no vision_config) whose AutoProcessor returns a
    tokenizer must be returned unchanged."""
    config = types.SimpleNamespace(model_type="qwen2")
    tokenizer = _FakeTokenizer()

    with mock.patch.object(wwb.AutoConfig, "from_pretrained", return_value=config), \
         mock.patch.object(wwb.AutoProcessor, "from_pretrained", return_value=tokenizer), \
         mock.patch.object(wwb.AutoImageProcessor, "from_pretrained") as img_proc:
        preprocessor, _ = wwb.load_processor(_args("dummy-text"))

    assert preprocessor is tokenizer
    img_proc.assert_not_called()
