from pathlib import Path
from typing import Any

import torch
from transformers import Qwen3OmniForConditionalGeneration, Qwen3OmniProcessor

# Requires transformers from:
# pip install git+https://github.com/huggingface/transformers@3d1a4f5e34753e51cb85052539c6ef10cab9a5c1

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def load_model(
    model_path: str | Path,
    device: str = "cpu",
    enable_audio: bool = True,
) -> tuple[Qwen3OmniForConditionalGeneration, Any]:
    processor = Qwen3OmniProcessor.from_pretrained(model_path)

    model = Qwen3OmniForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map=device,
        enable_audio_output=enable_audio,
    )
    model.eval()

    return model, processor
