# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native export of Molmo2 image-text-to-text models (allenai/MolmoWeb-4B).

The model is split into the three OpenVINO models consumed by ov::genai::VLMPipeline:

* ``openvino_text_embeddings_model``: ``input_ids`` -> ``inputs_embeds``.
* ``openvino_vision_embeddings_model``: the vision backbone (ViT, attention pooling and projector),
  ``images`` [batch, crops, patches, pixels] and ``pooled_patches_idx`` [batch, tokens, pool] -> pooled image
  features ``last_hidden_state`` [tokens, hidden]. The data-dependent crop / pooling-index preparation runs in
  GenAI's image preprocessing.
* ``openvino_language_model``: the stateful decoder taking ``inputs_embeds``, ``attention_mask``, ``position_ids``,
  ``token_type_ids`` and ``beam_idx``. Image tokens (token_type_ids == 1) attend to each other bidirectionally, so the
  attention mask is built inside the model.
"""

import gc
import json
import logging
from pathlib import Path
from typing import List, Optional

from openvino_genai._cli.exporters.base import NativeExporter
from openvino_genai._cli.exporters.common import (
    INT8_SYM_CONFIG,
    convert_exported_program,
    convert_tokenizer,
    copy_model_files,
    make_stateful,
    resolve_model_dir,
    save_submodel,
    torch_export,
    weight_compression_config_from_args,
)

logger = logging.getLogger(__name__)

# Checkpoint files needed for the export; MolmoWeb repositories also hold large training checkpoints.
_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.py", "*.jinja", "*.txt"]


def _processor_chat_template(model_dir: Path) -> Optional[str]:
    if (model_dir / "chat_template.jinja").is_file():
        return (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
    if (model_dir / "chat_template.json").is_file():
        return json.loads((model_dir / "chat_template.json").read_text(encoding="utf-8")).get("chat_template")
    return None


def _build_export_modules():
    import torch

    class TextEmbeddings(torch.nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self.embedding = embedding

        def forward(self, input_ids):
            return self.embedding(input_ids)

    class VisionEmbeddings(torch.nn.Module):
        def __init__(self, vision_backbone):
            super().__init__()
            self.vision_backbone = vision_backbone

        def forward(self, images, pooled_patches_idx):
            return self.vision_backbone(images, pooled_patches_idx)

    class KVCache:
        """Minimal cache built from the flat past key/value inputs; collects the updated tensors as outputs."""

        def __init__(self, keys: List[torch.Tensor], values: List[torch.Tensor]):
            self.keys = list(keys)
            self.values = list(values)

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], key_states], dim=-2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], value_states], dim=-2)
            return self.keys[layer_idx], self.values[layer_idx]

        def get_seq_length(self, layer_idx: int = 0) -> int:
            return self.keys[layer_idx].shape[-2]

    class LanguageModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.transformer = model.model.transformer
            self.lm_head = model.lm_head

        def forward(self, inputs_embeds, attention_mask, position_ids, token_type_ids, past_key_values):
            cache = KVCache(past_key_values[0::2], past_key_values[1::2])
            batch, seq_len = token_type_ids.shape
            past_len = past_key_values[0].shape[-2]

            # Causal attention over the cache and the new tokens, plus bidirectional attention between the image
            # tokens of the new chunk (Molmo2 builds it with an `or` mask function over token_type_ids at prefill).
            q_positions = torch.arange(seq_len, device=inputs_embeds.device) + past_len
            kv_positions = torch.arange(attention_mask.shape[1], device=inputs_embeds.device)
            causal = kv_positions[None, :] <= q_positions[:, None]
            q_is_image = token_type_ids == 1
            kv_is_image = torch.cat([q_is_image.new_zeros((batch, past_len)), q_is_image], dim=1)
            bidirectional = q_is_image[:, :, None] & kv_is_image[:, None, :]
            allowed = (causal[None] | bidirectional) & attention_mask[:, None, :].bool()
            mask = torch.where(allowed[:, None], 0.0, torch.finfo(inputs_embeds.dtype).min).to(inputs_embeds.dtype)

            outputs = self.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(past_len, past_len + seq_len, device=inputs_embeds.device),
            )
            logits = self.lm_head(outputs.last_hidden_state)
            return (logits, *[t for kv in zip(cache.keys, cache.values) for t in kv])

    return TextEmbeddings, VisionEmbeddings, LanguageModel


class Molmo2Exporter(NativeExporter):
    MODEL_TYPE = "molmo2"
    TASKS = ("image-text-to-text",)
    # Version the Molmo2 remote code is written for (it relies on the transformers 4.57 cache / RoPE APIs).
    TRANSFORMERS_VERSION = "4.57.6"
    REQUIRES_REMOTE_CODE = True

    def export(self) -> None:
        import torch
        from torch.export import Dim
        from transformers import AutoModelForImageTextToText

        args = self.args
        model_dir = resolve_model_dir(args.model, args.cache_dir, _ALLOW_PATTERNS)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading {args.model}")
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            trust_remote_code=True,
            dtype=torch.float32,
            attn_implementation="sdpa",
            variant=args.variant,
        )
        model.eval()
        TextEmbeddings, VisionEmbeddings, LanguageModel = _build_export_modules()
        dynamic = Dim.DYNAMIC
        text_config = model.config.text_config
        vit_config = model.config.vit_config

        logger.info("Exporting the text embeddings model")
        ep = torch_export(
            TextEmbeddings(model.model.transformer.wte),
            args=(torch.randint(0, 100, (2, 5)),),
            dynamic_shapes={"input_ids": {0: dynamic, 1: dynamic}},
        )
        text_embeddings = convert_exported_program(ep, ["input_ids"], ["inputs_embeds"])

        logger.info("Exporting the vision embeddings model")
        num_crops, num_patches = 3, vit_config.image_num_pos
        pixels_per_patch = vit_config.image_patch_size**2 * 3
        ep = torch_export(
            VisionEmbeddings(model.model.vision_backbone),
            args=(
                torch.rand(2, num_crops, num_patches, pixels_per_patch),
                torch.randint(0, num_crops * num_patches, (2, 7, 4)),
            ),
            dynamic_shapes={"images": {0: dynamic, 1: dynamic}, "pooled_patches_idx": {0: dynamic, 1: dynamic}},
        )
        vision_embeddings = convert_exported_program(ep, ["images", "pooled_patches_idx"], ["last_hidden_state"])

        logger.info("Exporting the language model")
        batch, seq_len, past_len = 2, 5, 3
        num_layers = text_config.num_hidden_layers
        kv_shape = (batch, text_config.num_key_value_heads, past_len, text_config.head_dim)
        past_key_values = [torch.rand(kv_shape) for _ in range(2 * num_layers)]
        ep = torch_export(
            LanguageModel(model),
            args=(
                torch.rand(batch, seq_len, text_config.hidden_size),
                torch.ones(batch, past_len + seq_len, dtype=torch.int64),
                torch.arange(past_len, past_len + seq_len).repeat(batch, 1),
                torch.zeros(batch, seq_len, dtype=torch.int64),
                past_key_values,
            ),
            dynamic_shapes={
                "inputs_embeds": {0: dynamic, 1: dynamic},
                "attention_mask": {0: dynamic, 1: dynamic},
                "position_ids": {0: dynamic, 1: dynamic},
                "token_type_ids": {0: dynamic, 1: dynamic},
                "past_key_values": [{0: dynamic, 2: dynamic}] * len(past_key_values),
            },
        )
        kv_names = [f"{layer}.{part}" for layer in range(num_layers) for part in ("key", "value")]
        language_model = convert_exported_program(
            ep,
            ["inputs_embeds", "attention_mask", "position_ids", "token_type_ids"]
            + [f"past_key_values.{name}" for name in kv_names],
            ["logits"] + [f"present.{name}" for name in kv_names],
        )
        if not args.disable_stateful:
            make_stateful(language_model)

        del ep, model
        gc.collect()

        compression = weight_compression_config_from_args(args)
        # As in optimum-intel, the requested compression applies to the language model, the other (small) models
        # get int8 weights.
        other_compression = INT8_SYM_CONFIG if compression is not None else None
        save_submodel(language_model, output / "openvino_language_model.xml", args.weight_format, compression)
        save_submodel(text_embeddings, output / "openvino_text_embeddings_model.xml", args.weight_format, other_compression)
        save_submodel(vision_embeddings, output / "openvino_vision_embeddings_model.xml", args.weight_format, other_compression)

        copy_model_files(model_dir, output)
        if not args.disable_convert_tokenizer:
            convert_tokenizer(model_dir, output, _processor_chat_template(model_dir))
        logger.info(f"Model exported to {output}")
