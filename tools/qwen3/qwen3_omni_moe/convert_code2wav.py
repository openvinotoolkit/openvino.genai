from pathlib import Path
import gc
from typing import Any

import torch
import openvino as ov
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
from transformers import Qwen3OmniMoeForConditionalGeneration

from .constants import CODE2WAV_NAME
from .utils import cleanup_torchscript_cache


def _diff_via_slice(
    input: torch.Tensor,
    n: int = 1,
    dim: int = -1,
    prepend: torch.Tensor | None = None,
    append: torch.Tensor | None = None,
) -> torch.Tensor:
    if prepend is not None:
        input = torch.cat([prepend, input], dim=dim)
    if append is not None:
        input = torch.cat([input, append], dim=dim)
    return input.narrow(dim, n, input.size(dim) - n) - input.narrow(dim, 0, input.size(dim) - n)


def convert_code2wav(model: Qwen3OmniMoeForConditionalGeneration, output_dir: Path) -> None:
    output_path = output_dir / CODE2WAV_NAME
    if output_path.exists():
        return

    print("Convert code2wav model...")
    code2wav = model.code2wav
    num_quantizers = code2wav.config.num_quantizers

    __make_16bit_traceable(code2wav)

    original_diff = torch.diff
    torch.diff = _diff_via_slice
    try:
        ov_model = ov.convert_model(
            code2wav,
            example_input={
                "codes": torch.ones([1, num_quantizers, 4], dtype=torch.long),
            },
        )
    finally:
        torch.diff = original_diff

    ov.save_model(ov_model, output_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print("Code2wav model successfully converted")
