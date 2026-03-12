from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration

from tools.qwen3.qwen3_omni_moe.compat import apply_dense_compat

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def load_model(
    model_path: str | Path,
    device: str = "auto",
    enable_audio: bool = True,
) -> tuple[Qwen3OmniMoeForConditionalGeneration, Any]:
    apply_dense_compat(model_path)

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        enable_audio_output=enable_audio,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor
