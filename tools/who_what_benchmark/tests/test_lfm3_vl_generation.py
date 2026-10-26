# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import openvino_genai

from whowhatbench.wwb import genai_gen_visual_text, genai_gen_visual_text_chat


@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("model_type", ["lfm3_vl", "qwen2_vl"])
def test_visual_text_thinking_mode(model_type, with_image):
    model = Mock(config=SimpleNamespace(model_type=model_type))
    model.generate.return_value = SimpleNamespace(texts=["answer"])
    image = np.zeros((16, 16, 3), dtype=np.uint8) if with_image else None
    answer = genai_gen_visual_text(model, "Describe", image, None, None, None, 16, True, None, None)
    assert answer == "answer"
    call = model.generate.call_args
    if model_type == "lfm3_vl":
        history = call.args[0]
        assert isinstance(history, openvino_genai.ChatHistory)
        assert history.get_extra_context() == {"enable_thinking": False}
        assert history.get_messages()[0]["content"] == "Describe"
        assert "image" not in call.kwargs
        if with_image:
            np.testing.assert_array_equal(call.kwargs["images"][0].data, image[None])
    else:
        assert call.args == ("Describe",)
        assert ("image" in call.kwargs) == with_image
    assert call.kwargs["max_new_tokens"] == 16
    assert call.kwargs["do_sample"] is False


@pytest.mark.parametrize("model_type", ["lfm3_vl", "qwen2_vl"])
def test_visual_chat_thinking_mode(model_type):
    contexts = []

    def generate(history, **kwargs):
        contexts.append(history.get_extra_context())
        return SimpleNamespace(texts=["answer"])

    model = Mock(config=SimpleNamespace(model_type=model_type))
    model.generate.side_effect = generate
    inputs = [{"prompt": prompt, "images": [], "videos": []} for prompt in ["Describe", "Explain"]]
    answers = genai_gen_visual_text_chat(model, inputs, None, None, 16, None, None)
    assert answers == ["answer", "answer"]
    expected = {"enable_thinking": False} if model_type == "lfm3_vl" else {}
    assert contexts == [expected, expected]
