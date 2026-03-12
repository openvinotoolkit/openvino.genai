from pathlib import Path
from typing import Any
import types

import torch
from torch.export import Dim
from transformers import Qwen3OmniMoeForConditionalGeneration
from .constants import (
    ATTENTION_MASK,
    CACHE_POSITION,
    HIDDEN_STATES,
    INPUTS_EMBEDS,
    LOGITS,
    POSITION_IDS,
    THINKER_LANGUAGE_NAME,
)
from .flat_cache import FlatCache
from .stateful_utils import patch_stateful
from .export_utils import export_to_ov
from .ov_model_utils import (
    quantize_and_save_ov_model,
    set_ov_model_names,
)


def _forward_wrap_thinker(
    self: Any,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: list[torch.Tensor] | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    cache_position: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, ...]:
    pkv_cache = None
    if past_key_values is not None:
        pkv_cache = FlatCache(past_key_values)

    outputs = self.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=pkv_cache,
        inputs_embeds=inputs_embeds,
        use_cache=pkv_cache is not None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        cache_position=cache_position,
    )
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    result = (logits, hidden_states)
    if pkv_cache is not None:
        for k, v in outputs["past_key_values"]:
            result = result + (k, v)
    return result


def convert_thinker_language(
    model: Qwen3OmniMoeForConditionalGeneration,
    output_dir: Path,
    quantization_config: dict[str, Any] | None = None,
) -> None:
    output_path = output_dir / THINKER_LANGUAGE_NAME
    if output_path.exists():
        return

    print("Convert thinker language model...")
    lang_model = model.thinker
    text_config = lang_model.model.config
    hidden_size = text_config.hidden_size
    num_pkv = text_config.num_hidden_layers
    num_kv_heads = text_config.num_key_value_heads
    head_dim = text_config.head_dim

    lang_model.forward = types.MethodType(_forward_wrap_thinker, lang_model)

    pkv_shape = (2, num_kv_heads, 2, head_dim)
    past_key_values = [torch.randn(pkv_shape) for _ in range(num_pkv * 2)]

    example_input = {
        INPUTS_EMBEDS: torch.randn(2, 2, hidden_size),
        ATTENTION_MASK: torch.ones([2, 4], dtype=torch.long),
        POSITION_IDS: torch.arange(2, 4).view(1, 1, -1).expand(4, 2, -1),
        "past_key_values": past_key_values,
        CACHE_POSITION: torch.arange(2, 4),
    }

    batch = Dim("batch")
    seq = Dim("seq")
    past_seq = Dim("past_seq")
    total_seq = Dim("total_seq", min=4)

    pkv_dyn = {0: batch, 2: past_seq}
    dynamic_shapes = {
        INPUTS_EMBEDS: {0: batch, 1: seq},
        ATTENTION_MASK: {0: batch, 1: total_seq},
        POSITION_IDS: {1: batch, 2: seq},
        "past_key_values": [pkv_dyn] * (num_pkv * 2),
        CACHE_POSITION: {0: seq},
    }

    ov_model = export_to_ov(lang_model, example_input, dynamic_shapes)
    set_ov_model_names(
        ov_model,
        [INPUTS_EMBEDS, ATTENTION_MASK, POSITION_IDS],
        [LOGITS, HIDDEN_STATES],
        num_pkv,
        suffix_input_names=[CACHE_POSITION],
    )
    patch_stateful(ov_model, 2)
    print("Thinker language model successfully converted")
    quantize_and_save_ov_model(ov_model, output_path, quantization_config)
    print("Thinker language model saved")
