# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native export of FlashVSR video super-resolution models (JunhaoZhuang/FlashVSR).

The model is split into the three OpenVINO models consumed by ov::genai::VideoToVideoPipeline, each in its own
directory with a config.json, next to a model_index.json:

* ``lq_projector``: ``lq_frames`` [1, 3, 4n, H, W] (bicubic x4 upscaled low-resolution frames in [-1, 1]) ->
  ``lq_latents`` [1, n, H/16, W/16, dim], one latent frame of conditioning features per 4 frames.
* ``transformer``: one streaming step of the Wan2.1 DiT over a chunk of latent frames, ``latents`` [1, 16, f, H/8, W/8],
  ``lq_latents``, ``frame_offset`` [] and ``top_k`` [] -> ``noise_pred``. The keys / values of the last three latent
  frame pairs of every block are model states. The self-attention is opset17 SparseAttention in its
  locality-constrained top-k mode.
* ``vae_decoder``: the tiny conditional decoder, ``latents`` [1, 16, T, H/8, W/8] and ``lq_frames`` [1, 3, 4T, H, W]
  -> ``frames`` [1, 3, 4T, H, W] in about [-1, 1]. The memories of its causal blocks are model states.

The pipeline resets the states at the start of every video.
"""

import gc
import json
import logging
from pathlib import Path

import numpy as np
import openvino as ov

from openvino_genai._cli.exporters.base import NativeExporter, register_exporter
from openvino_genai._cli.exporters.common import (
    INT8_SYM_CONFIG,
    convert_exported_program,
    resolve_model_dir,
    save_submodel,
    torch_export,
    weight_compression_config_from_args,
)

logger = logging.getLogger(__name__)

_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.ckpt", "README.md"]
_DIT_WEIGHTS = "diffusion_pytorch_model_streaming_dmd.safetensors"
_LQ_WEIGHTS = "LQ_proj_in.ckpt"
_DECODER_WEIGHTS = "TCDecoder.ckpt"
# The fixed text context of the one-step generator ([1, 512, 4096] T5 embedding of the FlashVSR prompt). It is part of
# the inference code of https://github.com/OpenImagingLab/FlashVSR (examples/WanVSR/prompt_tensor/posi_prompt.pth), not
# of the Hugging Face repository; it is read as a plain tensor file.
_CONTEXT_FILE = "posi_prompt.pth"
_CONTEXT_URL = (
    "https://raw.githubusercontent.com/OpenImagingLab/FlashVSR/main/examples/WanVSR/prompt_tensor/posi_prompt.pth"
)
# Side of the window of the locality-constrained sparse attention, in blocks (upstream local_range, 9 or 11).
_LOCAL_WINDOW = 11


def _load_context(model_dir: Path, cache_dir: str | None):
    import torch

    path = model_dir / _CONTEXT_FILE
    if not path.is_file():
        import urllib.request

        cache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "openvino_genai"
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"flashvsr_{_CONTEXT_FILE}"
        if not path.is_file():
            logger.info(f"Downloading the FlashVSR text context from {_CONTEXT_URL}")
            urllib.request.urlretrieve(_CONTEXT_URL, path)  # nosec B310
    return torch.load(path, map_location="cpu", weights_only=True)


def _lcsa_conversion_extension():
    """Converts the flashvsr::lcsa node of the exported program to opset17 SparseAttention (local top-k mode)."""
    from openvino import opset13, opset17
    from openvino.frontend import ConversionExtension

    def convert(context):
        q, k, v = context.get_input(0), context.get_input(1), context.get_input(2)
        grid = [opset13.convert(opset13.reshape(context.get_input(i), [1], False), "i64") for i in (3, 4)]
        top_k = opset13.convert(context.get_input(5), "i64")
        window = context.get_values_from_const_input(6, None, int)
        from openvino_genai._cli.exporters.flashvsr.modeling import BLOCK_SIZE

        node = opset17.sparse_attention(
            q,
            k,
            v,
            opset13.concat(grid, 0),
            top_k,
            key_block_size=BLOCK_SIZE,
            query_block_size=BLOCK_SIZE,
            causal=False,
            local_window=window,
        )
        return [node.output(0)]

    return ConversionExtension("flashvsr.lcsa.default", convert)


def _make_stateful(ov_model: ov.Model, state_pairs: dict[str, str], initializer) -> None:
    """Turns the given input -> output pairs into model states; ``initializer(input_name)`` gives the subgraph a state
    starts from."""
    from openvino._offline_transformations import apply_make_stateful_transformation

    apply_make_stateful_transformation(ov_model, state_pairs)
    # The variable id of a state is the input name followed by the output name.
    input_of_variable = {name + output: name for name, output in state_pairs.items()}
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            op.set_arguments([initializer(input_of_variable[op.get_variable_id()])])
    ov_model.validate_nodes_and_infer_types()


def _export_transformer(model_dir: Path, args, output: Path, compression) -> dict:
    import torch
    from openvino import opset13
    from torch.export import Dim

    from openvino_genai._cli.exporters.flashvsr import modeling

    logger.info("Exporting the transformer")
    wan = modeling.load_wan_transformer(model_dir)
    context = _load_context(model_dir, args.cache_dir)
    dit = modeling.FlashVSRTransformer(wan, context, _LOCAL_WINDOW, trim_cache=False).eval()
    num_layers, heads, head_dim = len(wan.blocks), dit.num_heads, dit.head_dim

    frame_pairs, grid_h, grid_w = (
        Dim("frame_pairs", min=1, max=64),
        Dim("grid_h", min=1, max=64),
        Dim("grid_w", min=1, max=64),
    )
    past_len = Dim("past_len", min=0)
    f, gh, gw = 4, 2, 3
    latents = torch.randn(1, 16, f, 16 * gh, 16 * gw)
    lq_latents = torch.randn(1, f, 8 * gh, 8 * gw, dit.dim)
    past = [
        torch.randn(1, heads, modeling.KV_CACHE_SLOTS * gh * gw * modeling.BLOCK_SIZE, head_dim)
        for _ in range(2 * num_layers)
    ]
    lat_dims = {2: 2 * frame_pairs, 3: 16 * grid_h, 4: 16 * grid_w}
    ep = torch_export(
        dit,
        args=(latents, lq_latents, torch.tensor(0), torch.tensor(100), past),
        dynamic_shapes={
            "latents": lat_dims,
            "lq_latents": {1: 2 * frame_pairs, 2: 8 * grid_h, 3: 8 * grid_w},
            "frame_offset": None,
            "top_k": None,
            "past_key_values": [{2: past_len}] * len(past),
        },
    )
    kv_names = [f"{layer}.{part}" for layer in range(num_layers) for part in ("key", "value")]
    ep = ep.run_decompositions(decomp_table={})
    ov_model = ov.convert_model(ep, extension=[_lcsa_conversion_extension()])
    names_in = ["latents", "lq_latents", "frame_offset", "top_k"] + [f"past_key_values.{n}" for n in kv_names]
    names_out = ["noise_pred"] + [f"present.{n}" for n in kv_names]
    for port, name in zip(ov_model.inputs, names_in):
        port.get_tensor().set_names({name})
    for port, name in zip(ov_model.outputs, names_out):
        port.get_tensor().set_names({name})
    del ep, dit, wan
    gc.collect()

    # Every attention reads the cached keys / values of the last KV_CACHE_SLOTS frame pairs (KV_CACHE_SLOTS * 2 * (H/16)
    # * (W/16) tokens) and the chunk's own ones. The state stores the attention's whole key / value input
    # concat(past[-keep:], current) and the next chunk trims it before its concat: one copy of the cache per layer
    # instead of the two of trimming the concatenation (Slice -> Assign).
    lat = ov_model.input("latents")
    shape = opset13.shape_of(lat, "i64")
    hw = opset13.multiply(
        opset13.divide(
            opset13.gather(shape, opset13.constant([3]), opset13.constant(0)), opset13.constant(np.int64(2))
        ),
        opset13.divide(
            opset13.gather(shape, opset13.constant([4]), opset13.constant(0)), opset13.constant(np.int64(2))
        ),
    )
    keep = opset13.multiply(hw, opset13.constant(np.int64(2 * modeling.KV_CACHE_SLOTS)))
    start = opset13.negative(keep)
    stop = opset13.constant(np.array([np.iinfo(np.int64).max], dtype=np.int64))
    one = opset13.constant(np.array([1], dtype=np.int64))
    axis = opset13.constant(np.array([2], dtype=np.int64))
    for name in names_in[4:]:
        past = ov_model.input(name)
        consumers = past.get_target_inputs()
        trimmed = opset13.slice(past, start, stop, one, axis)
        for consumer in consumers:
            consumer.replace_source_output(trimmed.output(0))
    ov_model.validate_nodes_and_infer_types()

    zero_kv = opset13.broadcast(
        opset13.constant(np.float32(0)), opset13.constant(np.array([1, heads, 0, head_dim], dtype=np.int64))
    )
    _make_stateful(ov_model, {f"past_key_values.{n}": f"present.{n}" for n in kv_names}, lambda name: zero_kv.output(0))
    save_submodel(ov_model, output / "transformer" / "openvino_model.xml", args.weight_format, compression)
    del ov_model
    gc.collect()
    return {
        "num_layers": num_layers,
        "num_attention_heads": heads,
        "attention_head_dim": head_dim,
        "in_channels": 16,
        "patch_size": [1, 2, 2],
        "block_size": [modeling.WINDOW[0], modeling.WINDOW[1], modeling.WINDOW[2]],
        "kv_cache_frame_pairs": modeling.KV_CACHE_SLOTS,
        "local_window": _LOCAL_WINDOW,
        "sparse_reference_area": modeling.SPARSE_REFERENCE_AREA,
        "first_chunk_frames": 2 * modeling.KV_CACHE_SLOTS,
        "chunk_frames": modeling.WINDOW[0],
    }


def _export_lq_projector(model_dir: Path, args, output: Path, compression) -> dict:
    import torch
    from torch.export import Dim

    from openvino_genai._cli.exporters.flashvsr import modeling

    logger.info("Exporting the LQ projector")
    lq_proj = modeling.LQProjector.from_checkpoint(model_dir / _LQ_WEIGHTS)
    clips, rows, cols = Dim("clips", min=1, max=256), Dim("rows", min=1, max=512), Dim("cols", min=1, max=512)
    ep = torch_export(
        lq_proj,
        args=(torch.rand(1, 3, 8, 32, 48) * 2 - 1,),
        dynamic_shapes={"lq_frames": {2: 4 * clips, 3: 16 * rows, 4: 16 * cols}},
    )
    ov_model = convert_exported_program(ep, ["lq_frames"], ["lq_latents"])
    save_submodel(ov_model, output / "lq_projector" / "openvino_model.xml", args.weight_format, compression)
    del ov_model, ep, lq_proj
    gc.collect()
    return {"frames_per_latent": 4, "patch_size": modeling.LQ_PATCH, "out_channels": 1536, "causal": False}


def _export_decoder(model_dir: Path, args, output: Path, compression) -> dict:
    import torch
    from openvino import opset13
    from torch import nn
    from torch.export import Dim

    from openvino_genai._cli.exporters.flashvsr import modeling

    logger.info("Exporting the decoder")
    decoder = modeling.TCDecoder.from_checkpoint(model_dir / _DECODER_WEIGHTS)

    class Decoder(nn.Module):
        """[1, 16, T, h, w] latents and [1, 3, 4T, 8h, 8w] LQ frames -> [1, 3, 4T, 8h, 8w] frames in [-1, 1]."""

        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, latents, lq_frames, memories):
            cond = modeling.pixel_unshuffle_cond(lq_frames)
            frames, new_memories = self.decoder(latents[0].transpose(0, 1), cond, memories)
            return frames.transpose(0, 1).unsqueeze(0) * 2 - 1, new_memories

    t, h, w = Dim("frames", min=1, max=64), Dim("rows", min=1, max=1024), Dim("cols", min=1, max=1024)
    T, hs, ws = 3, 4, 6
    shapes = decoder.memory_shapes(hs, ws)
    ep = torch_export(
        Decoder(decoder),
        args=(torch.randn(1, 16, T, hs, ws), torch.rand(1, 3, 4 * T, 8 * hs, 8 * ws), [torch.randn(s) for s in shapes]),
        dynamic_shapes={
            "latents": {2: t, 3: h, 4: w},
            "lq_frames": {2: 4 * t, 3: 8 * h, 4: 8 * w},
            "memories": [{2: h * (1 << (i // 3)), 3: w * (1 << (i // 3))} for i in range(len(shapes))],
        },
    )
    mem_in = [f"memory.{i}" for i in range(len(shapes))]
    mem_out = [f"present_memory.{i}" for i in range(len(shapes))]
    ov_model = convert_exported_program(ep, ["latents", "lq_frames"] + mem_in, ["frames"] + mem_out)
    del ep
    # The memories start at zero: [1, C, h << level, w << level] of the latent size.
    lat_shape = opset13.shape_of(ov_model.input("latents"), "i64")
    lat_h = opset13.gather(lat_shape, opset13.constant([3]), opset13.constant(0))
    lat_w = opset13.gather(lat_shape, opset13.constant([4]), opset13.constant(0))
    initializers = {}
    for i, s in enumerate(shapes):
        scale = opset13.constant(np.array([1 << (i // 3)], dtype=np.int64))
        dims = opset13.concat(
            [
                opset13.constant(np.array([1, s[1]], dtype=np.int64)),
                opset13.multiply(lat_h, scale),
                opset13.multiply(lat_w, scale),
            ],
            0,
        )
        initializers[f"memory.{i}"] = opset13.broadcast(opset13.constant(np.float32(0)), dims).output(0)
    _make_stateful(ov_model, dict(zip(mem_in, mem_out)), initializers.__getitem__)
    save_submodel(ov_model, output / "vae_decoder" / "openvino_model.xml", args.weight_format, compression)
    del ov_model, decoder
    gc.collect()
    return {"temporal_upscale": 4, "spatial_upscale": 8, "trim_first_frames": 3, "latent_channels": 16}


@register_exporter
class FlashVSRExporter(NativeExporter):
    MODEL_TYPE = "flashvsr"
    TASKS = ("video-to-video",)
    # FlashVSR does not use transformers; the DiT is loaded into the native diffusers WanTransformer3DModel.
    TRANSFORMERS_VERSION = None
    EXTRA_REQUIREMENTS = ("diffusers>=0.35",)

    @classmethod
    def validate_args(cls, args) -> None:
        super().validate_args(args)

    def export(self) -> None:
        import torch

        args = self.args
        model_dir = resolve_model_dir(args.model, args.cache_dir, _ALLOW_PATTERNS)
        output = Path(args.output)
        for sub in ("transformer", "lq_projector", "vae_decoder"):
            (output / sub).mkdir(parents=True, exist_ok=True)

        compression = weight_compression_config_from_args(args)
        # As for the other native exporters, the requested compression applies to the main model (the DiT), the other
        # models get int8_sym weights.
        other_compression = INT8_SYM_CONFIG if compression is not None else None
        with torch.no_grad():
            configs = {
                "transformer": _export_transformer(model_dir, args, output, compression),
                "lq_projector": _export_lq_projector(model_dir, args, output, other_compression),
                "vae_decoder": _export_decoder(model_dir, args, output, other_compression),
            }
        for sub, config in configs.items():
            (output / sub / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        model_index = {
            "_class_name": "FlashVSRPipeline",
            "scale_factor": 4,
            "transformer": ["flashvsr", "FlashVSRTransformer"],
            "lq_projector": ["flashvsr", "LQProjector"],
            "vae_decoder": ["flashvsr", "TCDecoder"],
        }
        (output / "model_index.json").write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")
        logger.info(f"Model exported to {output}")
