# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native export of BAGEL (ByteDance-Seed/BAGEL-7B-MoT) for ov::genai::BagelPipeline.

The layout is the one of the optimum-intel export (``optimum.exporters.openvino.model_configs.BagelOpenVINOConfig``),
five models without a batch dimension:

* ``openvino_language_model``: one stateless mixture-of-transformers decoder serving both experts. Understanding
  tokens (``und_index``) and VAE latent tokens (``gen_index``, embedded from ``x_t``, ``timestep`` and
  ``vae_position_ids``) share one attention; outputs are the text ``logits`` and the flow-matching velocity ``v_t``.
  The KV cache stays explicit inputs / outputs: BagelPipeline forks it for the classifier-free-guidance branches.
* ``openvino_text_embeddings_model``: ``input_ids`` -> ``inputs_embeds``.
* ``openvino_vision_embeddings_model``: SigLIP, connector and ViT position embedding of the patches of one image.
* ``openvino_vae_encoder_model`` / ``openvino_vae_decoder_model``: the latent autoencoder (posterior mean). The decoder
  has a dynamic batch, so the frames of ``BagelPipeline.generate_frames`` decode in one call on CPU and GPU.
"""

import gc
import logging
import tempfile
from pathlib import Path

import openvino as ov

from openvino_genai._cli.args import no_compression_parameter_provided
from openvino_genai._cli.exporters.base import NativeExporter, register_exporter
from openvino_genai._cli.exporters.common import (
    WeightCompressionConfig,
    convert_exported_program,
    convert_tokenizer,
    copy_model_files,
    processor_chat_template,
    resolve_compression,
    resolve_model_dir,
    save_submodel,
    torch_export,
    weight_compression_config_from_args,
)

logger = logging.getLogger(__name__)

_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.jinja", "*.txt"]

# The understanding residual stream of BAGEL-7B-MoT reaches ~1.6e5 (attention-sink tokens) from layer 3 on, above the
# f16 maximum, so f16 inference (GPU, NPU) overflows without rescaling. 16 brings the peak to ~1e4.
_UND_SCALE = 16.0
_UND_SCALE_FROM_LAYER = 3

# optimum-intel's default for `--weight-format int4` of ByteDance-Seed/BAGEL-7B-MoT
_DEFAULT_INT4_CONFIG = WeightCompressionConfig(
    mode="int4_sym", ratio=1.0, group_size=128, backup_precision="int8_sym", group_size_fallback="adjust"
)


def _und_scale(layer_idx: int) -> float:
    return _UND_SCALE if layer_idx >= _UND_SCALE_FROM_LAYER else 1.0


def _rescale_understanding_stream(text_model) -> list[float]:
    """Carries the understanding residual stream at 1 / _UND_SCALE from layer _UND_SCALE_FROM_LAYER on, as
    ``optimum.exporters.openvino.model_patcher.BagelLanguageModelPatcher`` does. The reparametrization is exact: the
    scale is folded into the understanding o_proj / down_proj weights, RMSNorm(x / s) with eps / s^2 equals RMSNorm(x),
    and power-of-two scaling is lossless. Returns the factor applied to the stream after the attention of each layer.
    """
    rescale = []
    for idx, layer in enumerate(text_model.layers):
        scale_in = _und_scale(idx - 1) if idx else 1.0
        scale_out = _und_scale(idx)
        layer.self_attn.o_proj.weight.data /= scale_in
        layer.mlp.down_proj.weight.data /= scale_out
        layer.input_layernorm.variance_epsilon /= scale_in**2
        layer.post_attention_layernorm.variance_epsilon /= scale_out**2
        rescale.append(scale_in / scale_out)
    text_model.norm.variance_epsilon /= _und_scale(len(text_model.layers) - 1) ** 2
    return rescale


def _build_export_modules():
    import torch
    import torch.nn.functional as F
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    class LanguageModel(torch.nn.Module):
        """Adapted from ``optimum.exporters.openvino.model_patcher.BagelLanguageModelForExport``."""

        def __init__(self, model, und_rescale: list[float]):
            super().__init__()
            self.bagel = model.model
            self.lm_head = model.lm_head
            self.und_rescale = und_rescale

        def forward(
            self,
            inputs_embeds,
            und_index,
            gen_index,
            x_t,
            timestep,
            vae_position_ids,
            attention_mask,
            position_ids,
            past_key_values,
        ):
            bagel = self.bagel
            text = bagel.language_model
            latents = bagel.vae2llm(x_t) + bagel.time_embedder(timestep)
            # [1, G] indices: NPUW moves constant-table lookups to the host and requires token-id shaped indices
            latents = latents + bagel.latent_pos_embed[vae_position_ids[None]][0]
            inverse = torch.argsort(torch.cat([und_index, gen_index]))
            hidden = torch.cat([inputs_embeds[und_index], latents], 0)[inverse]
            cos, sin = text.rotary_emb(hidden[None], position_ids[None])
            present = []
            for idx, layer in enumerate(text.layers):
                hidden, key, value = self._decoder_layer(
                    idx,
                    layer,
                    hidden,
                    und_index,
                    gen_index,
                    inverse,
                    cos,
                    sin,
                    attention_mask,
                    past_key_values[2 * idx],
                    past_key_values[2 * idx + 1],
                )
                present += [key, value]
            logits = self.lm_head(text.norm(hidden[und_index]))
            v_t = bagel.llm2vae(text.norm_moe_gen(hidden[gen_index]))
            return (logits, v_t, *present)

        def _decoder_layer(
            self, idx, layer, hidden, und_index, gen_index, inverse, cos, sin, attention_mask, past_key, past_value
        ):
            attn = layer.self_attn
            head_dim = attn.head_dim

            def project(x, q_proj, k_proj, v_proj, q_norm, k_norm):
                heads = (-1, head_dim)
                return (
                    q_norm(q_proj(x).unflatten(-1, heads)),
                    k_norm(k_proj(x).unflatten(-1, heads)),
                    v_proj(x).unflatten(-1, heads),
                )

            und, gen = hidden[und_index], hidden[gen_index]
            und_qkv = project(
                layer.input_layernorm(und), attn.q_proj, attn.k_proj, attn.v_proj, attn.q_norm, attn.k_norm
            )
            gen_qkv = project(
                layer.input_layernorm_moe_gen(gen),
                attn.q_proj_moe_gen,
                attn.k_proj_moe_gen,
                attn.v_proj_moe_gen,
                attn.q_norm_moe_gen,
                attn.k_norm_moe_gen,
            )
            query, key, value = (torch.cat([u, g], 0)[inverse].transpose(0, 1)[None] for u, g in zip(und_qkv, gen_qkv))
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            key = torch.cat([past_key, key], 2)
            value = torch.cat([past_value, value], 2)
            groups = attn.num_key_value_groups
            out = F.scaled_dot_product_attention(
                query,
                repeat_kv(key, groups),
                repeat_kv(value, groups),
                attn_mask=attention_mask,
                scale=attn.scaling,
            )
            out = out[0].transpose(0, 1).flatten(1)
            und = und + attn.o_proj(out[und_index])
            gen = gen + attn.o_proj_moe_gen(out[gen_index])
            if self.und_rescale[idx] != 1.0:
                und = und * self.und_rescale[idx]
            und = und + layer.mlp(layer.post_attention_layernorm(und))
            gen = gen + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(gen))
            return torch.cat([und, gen], 0)[inverse], key, value

    class TextEmbeddings(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.embed_tokens = model.get_input_embeddings()

        def forward(self, input_ids):
            return self.embed_tokens(input_ids)

    class VisionEmbeddings(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.bagel = model.model

        def forward(self, pixel_values, position_ids):
            return self.bagel.get_image_features(pixel_values, position_ids, return_dict=True).pooler_output

    class VAEEncoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.vae = model.model.vae

        def forward(self, pixel_values):
            return self.vae.encode(pixel_values, sample=False)

    class VAEDecoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.vae = model.model.vae

        def forward(self, latents):
            return self.vae.decode(latents)

    return LanguageModel, TextEmbeddings, VisionEmbeddings, VAEEncoder, VAEDecoder


@register_exporter
class BagelExporter(NativeExporter):
    MODEL_TYPE = "bagel"
    TASKS = ("image-text-to-text",)
    TRANSFORMERS_VERSION = "5.23.0.dev0"

    @classmethod
    def transformers_requirement(cls) -> str:
        # BAGEL is not in a transformers release yet
        return (
            "transformers @ https://github.com/padatta/transformers/archive/"
            "38b3e74123a005abfd071d2709bbbb017a7919e1.tar.gz"
        )

    def _compression(self) -> WeightCompressionConfig | None:
        args = self.args
        if args.weight_format == "int4" and no_compression_parameter_provided(args):
            return _DEFAULT_INT4_CONFIG
        return weight_compression_config_from_args(args)

    def export(self) -> None:
        import torch
        from torch.export import Dim
        from transformers import BagelForConditionalGeneration

        args = self.args
        model_dir = resolve_model_dir(args.model, args.cache_dir, _ALLOW_PATTERNS)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading {args.model}")
        model = BagelForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32, variant=args.variant)
        model.eval()
        LanguageModel, TextEmbeddings, VisionEmbeddings, VAEEncoder, VAEDecoder = _build_export_modules()
        dynamic = Dim.DYNAMIC
        config = model.config
        text_config = config.text_config

        compression = self._compression()
        # As optimum-intel, the requested compression applies to the language model, the other parts stay in FP16
        other_format = "fp16" if compression is not None else args.weight_format

        logger.info("Exporting the text embeddings model")
        ep = torch_export(TextEmbeddings(model), args=(torch.arange(6),), dynamic_shapes={"input_ids": {0: dynamic}})
        save_submodel(
            convert_exported_program(ep, ["input_ids"], ["inputs_embeds"]),
            output / "openvino_text_embeddings_model.xml",
            other_format,
            None,
        )

        logger.info("Exporting the vision embeddings model")
        vision_config = config.vision_config
        num_patches = 16
        ep = torch_export(
            VisionEmbeddings(model),
            args=(
                torch.randn(num_patches, vision_config.num_channels * vision_config.patch_size**2),
                torch.arange(num_patches),
            ),
            dynamic_shapes={"pixel_values": {0: dynamic}, "position_ids": {0: dynamic}},
        )
        save_submodel(
            convert_exported_program(ep, ["pixel_values", "position_ids"], ["image_embeds"]),
            output / "openvino_vision_embeddings_model.xml",
            other_format,
            None,
        )

        latent_size = 4
        image_size = latent_size * 2 ** (len(config.vae_config.channel_multiplier) - 1)
        logger.info("Exporting the VAE encoder model")
        ep = torch_export(
            VAEEncoder(model),
            args=(torch.rand(1, config.vae_config.in_channels, image_size, image_size) * 2 - 1,),
            dynamic_shapes={"pixel_values": {2: dynamic, 3: dynamic}},
        )
        save_submodel(
            convert_exported_program(ep, ["pixel_values"], ["latent"]),
            output / "openvino_vae_encoder_model.xml",
            other_format,
            None,
        )

        logger.info("Exporting the VAE decoder model")
        # a batch of 2 keeps the batch dimension out of the traced shapes
        ep = torch_export(
            VAEDecoder(model),
            args=(torch.randn(2, config.vae_config.latent_channels, latent_size, latent_size),),
            dynamic_shapes={"latents": {0: dynamic, 2: dynamic, 3: dynamic}},
        )
        save_submodel(
            convert_exported_program(ep, ["latents"], ["image"]),
            output / "openvino_vae_decoder_model.xml",
            other_format,
            None,
        )

        logger.info("Exporting the language model")
        text_model = model.model.language_model
        und_rescale = _rescale_understanding_stream(text_model)
        # a cached prefix followed by one image block (start, latent tokens, end): both experts, the bidirectional
        # block and the past / present concatenation are traced
        num_latents, past_len = 4, 3
        seq_len = num_latents + 2
        head_dim = getattr(text_config, "head_dim", None) or text_config.hidden_size // text_config.num_attention_heads
        patch_dim = config.latent_patch_size**2 * config.vae_config.latent_channels
        num_layers = text_config.num_hidden_layers
        past_key_values = [
            torch.zeros(1, text_config.num_key_value_heads, past_len, head_dim) for _ in range(2 * num_layers)
        ]
        ep = torch_export(
            LanguageModel(model, und_rescale),
            args=(
                torch.randn(seq_len, text_config.hidden_size),
                torch.tensor([0, seq_len - 1]),
                torch.arange(1, seq_len - 1),
                torch.randn(num_latents, patch_dim),
                torch.tensor([0.5]),
                torch.arange(num_latents),
                torch.zeros(1, 1, seq_len, past_len + seq_len),
                torch.full((seq_len,), past_len),
                past_key_values,
            ),
            dynamic_shapes={
                "inputs_embeds": {0: dynamic},
                "und_index": {0: dynamic},
                "gen_index": {0: dynamic},
                "x_t": {0: dynamic},
                "timestep": {},
                "vae_position_ids": {0: dynamic},
                "attention_mask": {2: dynamic, 3: dynamic},
                "position_ids": {0: dynamic},
                "past_key_values": [{2: dynamic}] * len(past_key_values),
            },
        )
        kv_names = [f"{layer}.{part}" for layer in range(num_layers) for part in ("key", "value")]
        language_model = convert_exported_program(
            ep,
            [
                "inputs_embeds",
                "und_index",
                "gen_index",
                "x_t",
                "timestep",
                "vae_position_ids",
                "attention_mask",
                "position_ids",
            ]
            + [f"past_key_values.{name}" for name in kv_names],
            ["logits", "v_t"] + [f"present.{name}" for name in kv_names],
        )
        generation_config = model.generation_config
        del ep, model, text_model
        gc.collect()

        # As in the Molmo2 exporter, the language model is compressed from a saved floating-point copy, so its weights
        # are memory-mapped instead of staying in process memory next to the compressed ones
        lm_path = output / "openvino_language_model.xml"
        language_compression = resolve_compression(language_model, args.weight_format, compression)
        if language_compression is None:
            save_submodel(language_model, lm_path, args.weight_format, None)
            del language_model
        else:
            with tempfile.TemporaryDirectory(dir=output, prefix=".genai_cli_") as tmp_dir:
                tmp_path = Path(tmp_dir) / lm_path.name
                ov.save_model(language_model, tmp_path, compress_to_fp16=False)
                del language_model
                gc.collect()
                save_submodel(ov.Core().read_model(tmp_path), lm_path, args.weight_format, language_compression)
        gc.collect()

        copy_model_files(model_dir, output)
        # BagelPipeline reads the EOS token from generation_config.json, which a checkpoint may not include
        generation_config.save_pretrained(output)
        if not args.disable_convert_tokenizer:
            convert_tokenizer(model_dir, output, processor_chat_template(model_dir))
        logger.info(f"Model exported to {output}")
