from pathlib import Path
from typing import Any
import types

import torch
from torch.export import Dim
from transformers import Qwen3OmniMoeForConditionalGeneration
from .constants import (
    ATTENTION_MASK,
    CODE_PREDICTOR_NAME,
    HIDDEN_STATES,
    INPUTS_EMBEDS,
    LOGITS,
)
from .flat_cache import FlatCache
from .stateful_utils import patch_stateful
from .export_utils import export_to_ov
from .ov_model_utils import (
    quantize_and_save_ov_model,
    set_ov_model_names,
)


def _forward_wrap_code_predictor(
    self: Any,
    attention_mask: torch.Tensor | None = None,
    past_key_values: list[torch.Tensor] | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
) -> tuple[torch.Tensor, ...]:
    pkv_cache = None
    if past_key_values is not None:
        pkv_cache = FlatCache(past_key_values)

    outputs = self.model(
        attention_mask=attention_mask,
        past_key_values=pkv_cache,
        inputs_embeds=inputs_embeds,
        use_cache=pkv_cache is not None,
    )
    hidden_states = outputs[0]
    logits = torch.stack([head(hidden_states).float() for head in self.lm_head], dim=0)

    result = (logits, hidden_states)
    if pkv_cache is not None:
        for k, v in outputs["past_key_values"]:
            result = result + (k, v)
    return result


def convert_code_predictor(
    model: Qwen3OmniMoeForConditionalGeneration,
    output_dir: Path,
    quantization_config: dict[str, Any] | None = None,
) -> None:
    output_path = output_dir / CODE_PREDICTOR_NAME
    if output_path.exists():
        return

    print("Convert code predictor model...")
    code_predictor = model.talker.code_predictor
    cp_config = code_predictor.config
    hidden_size = cp_config.hidden_size
    num_pkv = cp_config.num_hidden_layers
    num_kv_heads = cp_config.num_key_value_heads
    head_dim = cp_config.head_dim

    code_predictor.forward = types.MethodType(_forward_wrap_code_predictor, code_predictor)

    pkv_shape = (2, num_kv_heads, 2, head_dim)
    past_key_values = [torch.randn(pkv_shape) for _ in range(num_pkv * 2)]

    example_input = {
        INPUTS_EMBEDS: torch.randn(2, 2, hidden_size),
        ATTENTION_MASK: torch.ones([2, 4], dtype=torch.long),
        "past_key_values": past_key_values,
    }

    batch = Dim("batch")
    seq = Dim("seq")
    past_seq = Dim("past_seq")
    total_seq = Dim("total_seq", min=4)

    pkv_dyn = {0: batch, 2: past_seq}
    dynamic_shapes = {
        INPUTS_EMBEDS: {0: batch, 1: seq},
        ATTENTION_MASK: {0: batch, 1: total_seq},
        "past_key_values": [pkv_dyn] * (num_pkv * 2),
    }

    ov_model = export_to_ov(code_predictor, example_input, dynamic_shapes)
    set_ov_model_names(
        ov_model,
        [INPUTS_EMBEDS, ATTENTION_MASK],
        [LOGITS, HIDDEN_STATES],
        num_pkv,
    )
    patch_stateful(ov_model, 2)
    print("Code predictor model successfully converted")
    quantize_and_save_ov_model(ov_model, output_path, quantization_config)
    print("Code predictor model saved")
