# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Loads the diffusion model of a diffusers LTX-2 pipeline from a GGUF file of the original LTX-2 checkpoint, such as the
files of unsloth/LTX-2.3-GGUF. Those are made with ComfyUI-GGUF, which keeps the Lightricks tensor names, so the names
are mapped to diffusers ones the way diffusers' scripts/convert_ltx2_to_diffusers.py does it.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

# Original LTX-2 key fragments and their diffusers names, applied in this order: LTX_2_3_TRANSFORMER_KEYS_RENAME_DICT
# of diffusers' scripts/convert_ltx2_to_diffusers.py. The single-file converter built into diffusers only has the
# LTX-2.0 names, without the prompt AdaLN ones of LTX-2.3.
TRANSFORMER_KEYS_RENAME = (
    ("patchify_proj", "proj_in"),
    ("audio_patchify_proj", "audio_proj_in"),
    ("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift"),
    ("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate"),
    ("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift"),
    ("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate"),
    ("scale_shift_table_a2v_ca_video", "video_a2v_cross_attn_scale_shift_table"),
    ("scale_shift_table_a2v_ca_audio", "audio_a2v_cross_attn_scale_shift_table"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
    ("audio_prompt_adaln_single", "audio_prompt_adaln"),
    ("prompt_adaln_single", "prompt_adaln"),
)
# The connector rename dicts of LTX-2.0 and LTX-2.3 from the same script.
CONNECTORS_KEYS_RENAME = (
    ("connectors.", ""),
    ("video_embeddings_connector", "video_connector"),
    ("audio_embeddings_connector", "audio_connector"),
    ("transformer_1d_blocks", "transformer_blocks"),
    ("text_embedding_projection.aggregate_embed", "text_proj_in"),
    ("text_embedding_projection.audio_aggregate_embed", "audio_text_proj_in"),
    ("text_embedding_projection.video_aggregate_embed", "video_text_proj_in"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
)
# Keys of the text connectors, which are part of the original diffusion model but a separate `connectors` component of
# the diffusers pipeline.
CONNECTORS_KEY_PREFIXES = (
    "video_embeddings_connector",
    "audio_embeddings_connector",
    "transformer_1d_blocks",
    "text_embedding_projection",
    "connectors.",
    "video_connector",
    "audio_connector",
    "text_proj_in",
)


def _rename(key, renames):
    for old, new in renames:
        key = key.replace(old, new)
    return key


def convert_ltx2_state_dict(state_dict):
    """
    Splits a state dict of the LTX-2 diffusion model with the original names, with or without the
    "model.diffusion_model." prefix, into the state dicts of the diffusers `transformer` and `connectors`.
    """
    transformer, connectors = {}, {}
    for key, value in state_dict.items():
        key = key.removeprefix("model.diffusion_model.")
        if key.startswith(CONNECTORS_KEY_PREFIXES):
            connectors[_rename(key, CONNECTORS_KEYS_RENAME)] = value
            continue
        key = _rename(key, TRANSFORMER_KEYS_RENAME)
        # "adaln_single" is also part of other names, so only the top-level modules are renamed
        if key.startswith("adaln_single."):
            key = "time_embed." + key.removeprefix("adaln_single.")
        elif key.startswith("audio_adaln_single."):
            key = "audio_time_embed." + key.removeprefix("audio_adaln_single.")
        transformer[key] = value
    return transformer, connectors


def resolve_gguf_file(gguf_file):
    """A local path, or <org>/<repo>/<path in the repo> of a file on the Hugging Face Hub, which is downloaded."""
    if os.path.isfile(gguf_file):
        return gguf_file
    parts = gguf_file.split("/")
    if len(parts) < 3 or os.path.isabs(gguf_file) or gguf_file.startswith((".", "~")):
        raise FileNotFoundError(f"GGUF file {gguf_file} not found. Pass a local path or <org>/<repo>/<file>.gguf.")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id="/".join(parts[:2]), filename="/".join(parts[2:]))


def _check_keys(gguf_path, component, expected, loaded, optional=None):
    """Raises if the names in the file differ from the pipeline's. Names containing `optional` may be missing."""
    missing = sorted(k for k in set(expected) - set(loaded) if optional is None or optional not in k)
    unexpected = sorted(set(loaded) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"{gguf_path} does not match the {component} of the pipeline: {len(missing)} weights are missing "
            f"{missing[:5]}, {len(unexpected)} are unexpected {unexpected[:5]}."
        )


def load_ltx2_pipeline_from_gguf(pipeline_id, gguf_file, torch_dtype):
    """
    The diffusers LTX-2 pipeline `pipeline_id` with the transformer and text connector weights of the GGUF file.

    Quantized transformer weights stay quantized and are dequantized to `torch_dtype` when used, as diffusers does for
    GGUF checkpoints. The connectors cannot hold GGUF weights, so their weights from the file are dequantized to
    `torch_dtype` on load. The text projections of the connectors, which unsloth/LTX-2.3-GGUF ships in a separate
    file, and all other components are those of the pipeline.
    """
    from accelerate import init_empty_weights
    from diffusers import DiffusionPipeline, GGUFQuantizationConfig, LTX2VideoTransformer3DModel
    from diffusers.models.model_loading_utils import load_gguf_checkpoint
    from diffusers.pipelines.ltx2 import LTX2TextConnectors
    from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
    from gguf import GGUFReader

    transformer_class = (DiffusionPipeline.load_config(pipeline_id).get("transformer") or [None, None])[1]
    if transformer_class != LTX2VideoTransformer3DModel.__name__:
        raise NotImplementedError(
            f"--gguf-file is supported for LTX-2 pipelines only, "
            f"the transformer of {pipeline_id} is {transformer_class}."
        )
    gguf_path = resolve_gguf_file(gguf_file)

    # Check the names before the weights are read, on models without weights
    with init_empty_weights():
        empty_transformer = LTX2VideoTransformer3DModel.from_config(
            LTX2VideoTransformer3DModel.load_config(pipeline_id, subfolder="transformer")
        )
        empty_connectors = LTX2TextConnectors.from_config(
            LTX2TextConnectors.load_config(pipeline_id, subfolder="connectors")
        )
    transformer_names, connector_names = convert_ltx2_state_dict({t.name: None for t in GGUFReader(gguf_path).tensors})
    _check_keys(gguf_path, "transformer", empty_transformer.state_dict(), transformer_names)
    if connector_names:
        _check_keys(gguf_path, "connectors", empty_connectors.state_dict(), connector_names, optional="text_proj_in")
    else:
        logger.warning(f"{gguf_path} has no text connector weights, the connectors of {pipeline_id} are used")

    logger.info(f"Loading the transformer and text connector weights from {gguf_path}")
    transformer_sd, connectors_sd = convert_ltx2_state_dict(load_gguf_checkpoint(gguf_path))
    # GGUFLinear dequantizes the weights of Linear layers when they are used. Other tensors stored as BF16 or quantized
    # would be used as raw bytes, so they are dequantized now (ComfyUI-GGUF keeps them in F32 anyway).
    linear_weights = {
        f"{name}.weight" for name, m in empty_transformer.named_modules() if isinstance(m, torch.nn.Linear)
    }
    for key in [k for k, v in transformer_sd.items() if isinstance(v, GGUFParameter) and k not in linear_weights]:
        transformer_sd[key] = dequantize_gguf_tensor(transformer_sd[key]).to(torch_dtype)
    # The keys are diffusers names now, so from_single_file loads them as they are instead of applying its own mapping.
    transformer = LTX2VideoTransformer3DModel.from_single_file(
        transformer_sd,
        config=pipeline_id,
        subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
        torch_dtype=torch_dtype,
    )
    pipeline = DiffusionPipeline.from_pretrained(pipeline_id, transformer=transformer, torch_dtype=torch_dtype)

    # One tensor at a time, so that only one dequantized copy exists at once
    connector_params = pipeline.connectors.state_dict()
    with torch.no_grad():
        for key in list(connectors_sd):
            value = dequantize_gguf_tensor(connectors_sd.pop(key))
            if value.shape != connector_params[key].shape:
                raise ValueError(
                    f"{key} of {gguf_path} has the shape {tuple(value.shape)}, "
                    f"the connectors of the pipeline have {tuple(connector_params[key].shape)}."
                )
            connector_params[key].copy_(value)
    from_pipeline = sorted(connector_params.keys() - connector_names.keys())
    if connector_names and from_pipeline:
        logger.info(f"Connector weights not in {gguf_path}, used from {pipeline_id}: {from_pipeline}")
    return pipeline
