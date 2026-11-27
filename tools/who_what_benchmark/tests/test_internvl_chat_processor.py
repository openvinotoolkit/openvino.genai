# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

from whowhatbench import wwb


def test_load_processor_skips_native_processor_for_internvl_chat(monkeypatch):
    # Remote-code InternVL checkpoints may ship processor_config.json pointing to the native InternVLProcessor,
    # which cannot be built from their tokenizer. WWB must not instantiate any processor for them.
    config = SimpleNamespace(model_type="internvl_chat")
    auto_config = Mock(side_effect=[ValueError("remote code required"), config])
    auto_processor = Mock(side_effect=AttributeError("Qwen2TokenizerFast has no attribute start_image_token"))
    monkeypatch.setattr(wwb.AutoConfig, "from_pretrained", auto_config)
    monkeypatch.setattr(wwb.AutoProcessor, "from_pretrained", auto_processor)

    args = Namespace(base_model="dummy/internvl_chat", target_model=None, gguf_file=None, genai=False)
    processor, loaded_config = wwb.load_processor(args)

    assert processor is None
    assert loaded_config is config
    auto_processor.assert_not_called()
