from pathlib import Path
import gc

import torch
import openvino as ov
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
from transformers import Qwen3OmniMoeForConditionalGeneration

from .constants import THINKER_EMBEDDING_NAME
from .utils import cleanup_torchscript_cache


def convert_thinker_embedding(model: Qwen3OmniMoeForConditionalGeneration, output_dir: Path) -> None:
    output_path = output_dir / THINKER_EMBEDDING_NAME
    if output_path.exists():
        return

    print("Convert thinker embedding model...")
    embed_tokens = model.thinker.model.get_input_embeddings()
    __make_16bit_traceable(embed_tokens)

    ov_model = ov.convert_model(
        embed_tokens,
        example_input=torch.ones([2, 2], dtype=torch.int64),
    )
    ov.save_model(ov_model, output_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print("Thinker embedding model successfully converted")
