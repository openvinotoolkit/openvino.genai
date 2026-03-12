from pathlib import Path
from typing import Any
import gc
import types

import torch
import openvino as ov
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
from transformers import Qwen3OmniMoeForConditionalGeneration
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    apply_rotary_pos_emb_vision,
)

from .constants import VISION_MERGER_NAME, VISION_PATCHER_NAME
from .utils import cleanup_torchscript_cache


def _sdpa_attn_forward(
    self: Any,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q, k, v = qkv

    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
    q = q.squeeze(0).transpose(0, 1)
    k = k.squeeze(0).transpose(0, 1)
    v = v.transpose(0, 1)

    attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
    attn_output = attn_output.transpose(0, 1).reshape(seq_length, -1)
    return self.proj(attn_output)


def _block_forward(
    self: Any,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states),
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    return hidden_states


def _merger_forward(
    self: Any,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    deepstack_features: list[torch.Tensor] = []
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        if layer_num in self.deepstack_visual_indexes:
            idx = self.deepstack_visual_indexes.index(layer_num)
            deepstack_features.append(self.merger_list[idx](hidden_states))

    merged = self.merger(hidden_states)
    return (merged, *deepstack_features)


def convert_vision_patcher(model: Qwen3OmniMoeForConditionalGeneration, output_dir: Path) -> None:
    output_path = output_dir / VISION_PATCHER_NAME
    if output_path.exists():
        return

    print("Convert vision patcher model...")
    vision = model.thinker.visual
    __make_16bit_traceable(vision.patch_embed)

    temporal_patch_size = vision.patch_embed.temporal_patch_size
    patch_size = vision.patch_embed.patch_size
    in_channels = vision.patch_embed.in_channels
    input_dim = temporal_patch_size * patch_size * patch_size * in_channels

    ov_model = ov.convert_model(
        vision.patch_embed,
        example_input={"hidden_states": torch.randn([8, input_dim])},
    )
    ov.save_model(ov_model, output_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print("Vision patcher model successfully converted")


def convert_vision_merger(model: Qwen3OmniMoeForConditionalGeneration, output_dir: Path) -> None:
    output_path = output_dir / VISION_MERGER_NAME
    if output_path.exists():
        return

    print("Convert vision merger model...")
    vision = model.thinker.visual
    hidden_size = vision.config.hidden_size
    head_dim = hidden_size // vision.config.num_heads

    for block in vision.blocks:
        block.forward = types.MethodType(_block_forward, block)
        block.attn.forward = types.MethodType(_sdpa_attn_forward, block.attn)

    vision.forward = types.MethodType(_merger_forward, vision)
    __make_16bit_traceable(vision)

    seq_len = 8
    emb_dim = head_dim
    ov_model = ov.convert_model(
        vision,
        example_input={
            "hidden_states": torch.randn([seq_len, hidden_size], dtype=torch.float32),
            "attention_mask": torch.ones([1, seq_len, seq_len]),
            "position_embeddings": (
                torch.randn([seq_len, emb_dim]),
                torch.randn([seq_len, emb_dim]),
            ),
        },
    )
    ov.save_model(ov_model, output_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print("Vision merger model successfully converted")
