# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch modules of FlashVSR (JunhaoZhuang/FlashVSR) written for the export to OpenVINO.

FlashVSR is a one-step streaming video super-resolution model built of three parts:

* the LQ projector (``LQ_proj_in.ckpt``): pixel-unshuffle of the bicubic x4 upscaled low-resolution frames and two
  temporal convolutions; every 4 frames give one latent frame of features that is added to the input of the first DiT
  block;
* the Wan2.1-T2V-1.3B DiT (``diffusion_pytorch_model_streaming_dmd.safetensors``), which runs on chunks of latent
  frames with a key/value cache of the last three latent frame pairs and locality-constrained block-sparse
  self-attention;
* the tiny conditional decoder (``TCDecoder.ckpt``), a causal TAEHV-style decoder that gets the latents together with
  the pixel-unshuffled low-resolution frames.

The DiT weights are loaded into the native ``diffusers.WanTransformer3DModel``; there is no diffusers or transformers
class for the streaming forward, the LQ projector or the decoder, so they are implemented here. The modules follow the
inference code of https://github.com/OpenImagingLab/FlashVSR (FlashVSRTinyPipeline), which is only used as a
reference for the semantics and is never imported.
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# Every latent frame pair is cut into windows of 2 x 8 x 8 tokens; a window is one block of the block-sparse attention.
WINDOW = (2, 8, 8)
BLOCK_SIZE = WINDOW[0] * WINDOW[1] * WINDOW[2]
# The number of latent frame pairs kept in the key/value cache.
KV_CACHE_SLOTS = 3
# The LQ projector and the decoder see the low-resolution frames upscaled x4 and cut into 16 x 16 (8 x 8) patches.
LQ_PATCH = 16
# The sparse attention top-k budget is defined relative to a 768 x 1280 output (upstream topk_ratio).
SPARSE_REFERENCE_AREA = 768 * 1280
# The fixed timestep of the one-step generator.
TIMESTEP = 1000.0


# ---------------------------------------------------------------------------------------------------------------------
# Locality-constrained sparse attention
# ---------------------------------------------------------------------------------------------------------------------
def local_block_mask(grid_h: int, grid_w: int, window: int, device=None) -> torch.Tensor:
    """[G, G] bool: query block (ri, ci) may attend key block (rj, cj) of any frame pair when both offsets are inside
    the `window` x `window` window starting at ri - window // 2 (no clamping at the borders)."""
    r = torch.arange(grid_h, device=device).repeat_interleave(grid_w)
    c = torch.arange(grid_w, device=device).repeat(grid_h)
    start_r = r - window // 2
    start_c = c - window // 2
    in_row = (r[None, :] >= start_r[:, None]) & (r[None, :] <= start_r[:, None] + window - 1)
    in_col = (c[None, :] >= start_c[:, None]) & (c[None, :] <= start_c[:, None] + window - 1)
    return in_row & in_col


def sparse_top_k(grid_h: int, grid_w: int, sparse_ratio: float) -> int:
    """The number of (query block, key block) pairs every head keeps per query frame pair (upstream formula)."""
    blocks = grid_h * grid_w
    out_area = (grid_h * WINDOW[1] * LQ_PATCH) * (grid_w * WINDOW[2] * LQ_PATCH)
    topk_ratio = sparse_ratio * SPARSE_REFERENCE_AREA / out_area
    return int(blocks * blocks * topk_ratio) - 1


def select_blocks(q: torch.Tensor, k: torch.Tensor, grid_h: int, grid_w: int, top_k: int, window: int) -> torch.Tensor:
    """Block selection of the locality-constrained sparse attention.

    q: [N, H, L, E], k: [N, H, S, E], tokens grouped in blocks of BLOCK_SIZE, blocks ordered (frame pair, row, col).
    Returns the [N, H, L / BLOCK_SIZE, S / BLOCK_SIZE] bool mask of the kept block pairs.
    """
    n, h, length, e = q.shape
    nq, nk = length // BLOCK_SIZE, k.shape[2] // BLOCK_SIZE
    blocks = grid_h * grid_w
    q_slots = nq // blocks
    qb = q.float().view(n, h, nq, BLOCK_SIZE, e).mean(3)
    kb = k.float().view(n, h, nk, BLOCK_SIZE, e).mean(3)
    scores = qb @ kb.transpose(-1, -2) / math.sqrt(e)
    allowed = local_block_mask(grid_h, grid_w, window, q.device)
    allowed = allowed.repeat(nq // blocks, nk // blocks)
    scores = scores.masked_fill(~allowed, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    flat = probs.reshape(n, h, q_slots, blocks * nk)
    kth = min(flat.shape[-1] - 1, top_k)
    threshold = torch.topk(flat, kth + 1, dim=-1).values[..., -1:]
    return (flat > threshold).reshape(n, h, nq, nk)


def block_sparse_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, keep: torch.Tensor, rows_per_step: int = 16
) -> torch.Tensor:
    """Attention of every query block over the tokens of its kept key blocks; a query block without any kept key block
    gives zeros. q: [N, H, L, E], k/v: [N, H, S, E], keep: [N, H, L / B, S / B]."""
    n, h, length, e = q.shape
    nq = length // BLOCK_SIZE
    scale = 1.0 / math.sqrt(e)
    out = torch.zeros(n, h, length, v.shape[-1], dtype=torch.float32, device=q.device)
    kf, vf = k.float(), v.float()
    for start in range(0, nq, rows_per_step):
        stop = min(nq, start + rows_per_step)
        qs = q[:, :, start * BLOCK_SIZE : stop * BLOCK_SIZE].float()
        scores = qs @ kf.transpose(-1, -2) * scale
        token_keep = keep[:, :, start:stop].repeat_interleave(BLOCK_SIZE, 2).repeat_interleave(BLOCK_SIZE, 3)
        scores = scores.masked_fill(~token_keep, float("-inf"))
        probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        out[:, :, start * BLOCK_SIZE : stop * BLOCK_SIZE] = probs @ vf
    return out.to(q.dtype)


def lcsa_reference(q, k, v, grid_h: int, grid_w: int, top_k: int, window: int) -> torch.Tensor:
    keep = select_blocks(q, k, grid_h, grid_w, top_k, window)
    return block_sparse_attention(q, k, v, keep)


# torch.export keeps the attention as one node, which the OpenVINO conversion maps to opset17 SparseAttention in its
# locality-constrained top-k mode (see export.py).
@torch.library.custom_op("flashvsr::lcsa", mutates_args=())
def lcsa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, grid_h: int, grid_w: int, top_k: torch.Tensor, window: int
) -> torch.Tensor:
    return lcsa_reference(q, k, v, grid_h, grid_w, int(top_k), window)


@lcsa.register_fake
def _(q, k, v, grid_h, grid_w, top_k, window):
    return q.new_empty(q.shape[:-1] + v.shape[-1:])


# ---------------------------------------------------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------------------------------------------------
def sinusoidal_embedding(dim: int, position: torch.Tensor) -> torch.Tensor:
    sinusoid = torch.outer(
        position.double(), torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64).div(dim // 2))
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).float()


def rope_inv_freq(head_dim: int, theta: float = 10000.0) -> torch.Tensor:
    """[3, head_dim / 2] inverse frequencies of the frame / row / column axes, zero outside the axis' share
    (head_dim - 2 * (head_dim // 3), head_dim // 3, head_dim // 3 dimensions, as Wan 2.1)."""
    dims = (head_dim - 2 * (head_dim // 3), head_dim // 3, head_dim // 3)
    table = torch.zeros(3, head_dim // 2, dtype=torch.float64)
    offset = 0
    for axis, dim in enumerate(dims):
        table[axis, offset : offset + dim // 2] = 1.0 / theta ** (torch.arange(0, dim, 2)[: dim // 2].double() / dim)
        offset += dim // 2
    return table.float()


def window_partition(x: torch.Tensor, f: int, h: int, w: int) -> torch.Tensor:
    """[N, f*h*w, C] raster tokens -> [N, f*h*w, C] tokens grouped in windows, windows ordered (pair, row, col)."""
    n, _, c = x.shape
    wf, wh, ww = WINDOW
    x = x.reshape(n, f // wf, wf, h // wh, wh, w // ww, ww, c)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(n, f * h * w, c)


def window_reverse(x: torch.Tensor, f: int, h: int, w: int) -> torch.Tensor:
    n, _, c = x.shape
    wf, wh, ww = WINDOW
    x = x.reshape(n, f // wf, h // wh, w // ww, wf, wh, ww, c)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(n, f * h * w, c)


class FlashVSRTransformer(nn.Module):
    """One streaming step of the FlashVSR DiT over a chunk of latent frames.

    Inputs: ``latents`` [1, 16, f, H/8, W/8] (noise), ``lq_latents`` [1, f, H/16, W/16, dim] (LQ projector output),
    ``frame_offset`` [] (index of the first latent frame of the chunk, for RoPE), ``top_k`` [] (sparse attention
    budget, see sparse_top_k) and the past keys / values [1, heads, S, head_dim] of every block, tokens in window order.
    Outputs: ``noise_pred`` with the shape of ``latents`` and the new keys / values (the last KV_CACHE_SLOTS frame pairs).
    """

    def __init__(
        self, wan, context: torch.Tensor, local_window: int, use_custom_op: bool = True, trim_cache: bool = True
    ):
        super().__init__()
        self.wan = wan
        self.local_window = local_window
        self.use_custom_op = use_custom_op
        # The export returns the whole concatenated keys / values and trims them in the OpenVINO graph: the symbolic
        # slice start would need a guard on the dynamic cache length.
        self.trim_cache = trim_cache
        config = wan.config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.dim = self.num_heads * self.head_dim
        with torch.no_grad():
            emb = wan.condition_embedder
            t = emb.time_embedder(sinusoidal_embedding(config.freq_dim, torch.tensor([TIMESTEP])))
            t_mod = emb.time_proj(emb.act_fn(t)).unflatten(1, (6, self.dim))
            # The text context is fixed: the cross-attention keys / values of every block are constants.
            ctx = emb.text_embedder(context.float())
            cross_k = torch.stack([blk.attn2.norm_k(blk.attn2.to_k(ctx)) for blk in wan.blocks])
            cross_v = torch.stack([blk.attn2.to_v(ctx) for blk in wan.blocks])
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("t_mod", t_mod, persistent=False)
        # [layers, 1, heads, ctx_len, head_dim]
        heads = lambda x: x.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-2, -3)  # noqa: E731
        self.register_buffer("cross_k", heads(cross_k).contiguous(), persistent=False)
        self.register_buffer("cross_v", heads(cross_v).contiguous(), persistent=False)
        self.register_buffer("inv_freq", rope_inv_freq(self.head_dim), persistent=False)

    def rope(self, f: int, h: int, w: int, frame_offset: torch.Tensor):
        """cos / sin [f*h*w, 1, head_dim / 2] of the raster token positions (frame_offset + fi, hi, wi)."""
        fi = torch.arange(f, dtype=torch.float32) + frame_offset.to(torch.float32)
        hi = torch.arange(h, dtype=torch.float32)
        wi = torch.arange(w, dtype=torch.float32)
        pos = torch.stack(
            [
                fi.view(f, 1, 1).expand(f, h, w),
                hi.view(1, h, 1).expand(f, h, w),
                wi.view(1, 1, w).expand(f, h, w),
            ],
            dim=-1,
        ).reshape(f * h * w, 3)
        angles = pos @ self.inv_freq
        return torch.cos(angles).unsqueeze(1), torch.sin(angles).unsqueeze(1)

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x: [N, L, dim]; rotates the interleaved (even, odd) pairs of every head (Wan's complex multiplication)."""
        n, length, _ = x.shape
        x = x.float().reshape(n, length, self.num_heads, self.head_dim // 2, 2)
        x0, x1 = x[..., 0], x[..., 1]
        out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        return out.reshape(n, length, self.dim)

    def self_attention(self, blk, x, cos, sin, f, h, w, past_k, past_v, top_k):
        attn = blk.attn1
        n, length, _ = x.shape
        q = self.apply_rope(attn.norm_q(attn.to_q(x)), cos, sin)
        k = self.apply_rope(attn.norm_k(attn.to_k(x)), cos, sin)
        v = attn.to_v(x)
        heads = lambda t: t.reshape(n, length, self.num_heads, self.head_dim).transpose(1, 2)  # noqa: E731
        q = heads(window_partition(q, f, h, w))
        k = torch.cat([past_k, heads(window_partition(k, f, h, w))], dim=2)
        v = torch.cat([past_v, heads(window_partition(v, f, h, w))], dim=2)
        grid_h, grid_w = h // WINDOW[1], w // WINDOW[2]
        if self.use_custom_op:
            out = torch.ops.flashvsr.lcsa(q, k, v, grid_h, grid_w, top_k, self.local_window)
        else:
            out = lcsa_reference(q, k, v, grid_h, grid_w, int(top_k), self.local_window)
        out = window_reverse(out.transpose(1, 2).reshape(n, length, self.dim), f, h, w)
        if not self.trim_cache:
            return attn.to_out[0](out), k, v
        # The cache keeps the last KV_CACHE_SLOTS frame pairs; the first chunk has exactly that many.
        keep = KV_CACHE_SLOTS * grid_h * grid_w * BLOCK_SIZE
        total = k.shape[2]
        return attn.to_out[0](out), k[:, :, total - keep :], v[:, :, total - keep :]

    def cross_attention(self, blk, layer, x):
        attn = blk.attn2
        n, length, _ = x.shape
        q = attn.norm_q(attn.to_q(x)).reshape(n, length, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, self.cross_k[layer], self.cross_v[layer])
        return attn.to_out[0](out.transpose(1, 2).reshape(n, length, self.dim))

    def forward(self, latents, lq_latents, frame_offset, top_k, past_key_values):
        wan = self.wan
        x = wan.patch_embedding(latents)
        _, _, f, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + lq_latents.reshape(1, f * h * w, self.dim)
        cos, sin = self.rope(f, h, w, frame_offset)
        presents = []
        for layer, blk in enumerate(wan.blocks):
            shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = (blk.scale_shift_table + self.t_mod).chunk(
                6, dim=1
            )
            norm_x = blk.norm1(x.float()) * (1 + scale_msa) + shift_msa
            attn_out, k, v = self.self_attention(
                blk, norm_x, cos, sin, f, h, w, past_key_values[2 * layer], past_key_values[2 * layer + 1], top_k
            )
            presents += [k, v]
            x = x + attn_out * gate_msa
            x = x + self.cross_attention(blk, layer, blk.norm2(x.float()))
            norm_x = blk.norm3(x.float()) * (1 + c_scale) + c_shift
            x = x + blk.ffn(norm_x) * c_gate
        shift, scale = (wan.scale_shift_table + self.t.unsqueeze(1)).chunk(2, dim=1)
        x = wan.proj_out(wan.norm_out(x.float()) * (1 + scale) + shift)
        # Unpatchify [1, f*h*w, 16*1*2*2] -> [1, 16, f, 2h, 2w].
        pt, ph, pw = wan.config.patch_size
        x = x.reshape(1, f, h, w, pt, ph, pw, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(1, -1, f * pt, h * ph, w * pw), presents


def load_wan_transformer(model_dir: Path):
    """The DiT weights in the native diffusers WanTransformer3DModel (Wan2.1-T2V-1.3B geometry)."""
    from diffusers import WanTransformer3DModel
    from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
    from safetensors.torch import load_file

    state = load_file(str(model_dir / "diffusion_pytorch_model_streaming_dmd.safetensors"))
    dim = state["patch_embedding.weight"].shape[0]
    num_layers = len({key.split(".")[1] for key in state if key.startswith("blocks.")})
    wan = WanTransformer3DModel(
        patch_size=tuple(state["patch_embedding.weight"].shape[2:]),
        num_attention_heads=dim // 128,
        attention_head_dim=128,
        in_channels=state["patch_embedding.weight"].shape[1],
        out_channels=state["head.head.weight"].shape[0] // 4,
        text_dim=state["text_embedding.0.weight"].shape[1],
        freq_dim=state["time_embedding.0.weight"].shape[1],
        ffn_dim=state["blocks.0.ffn.0.weight"].shape[0],
        num_layers=num_layers,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
    )
    missing, unexpected = wan.load_state_dict(convert_wan_transformer_to_diffusers(state), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected FlashVSR DiT checkpoint: missing {missing}, unexpected {unexpected}")
    return wan.float().eval()


# ---------------------------------------------------------------------------------------------------------------------
# LQ projector
# ---------------------------------------------------------------------------------------------------------------------
class ChannelRMSNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scale = channels**0.5
        self.gamma = nn.Parameter(torch.ones(channels, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma


class LQProjector(nn.Module):
    """The LQ projector of FlashVSR v1 (``Buffer_LQ4x_Proj``).

    In v1 the temporal padding of both convolutions is the end of the clip itself, so every clip of 4 frames gives one
    latent frame of features independently of the others. Input: ``lq_frames`` [1, 3, 4n, H, W] in [-1, 1] at the
    output resolution; output [1, n, H/16, W/16, dim]. For v1.1 (``causal=True``, ``Causal_LQ4x_Proj``) the
    convolutions are causal over the clips instead and the model takes and returns their two-frame caches.
    """

    def __init__(self, out_dim: int = 1536, hidden1: int = 2048, hidden2: int = 3072, causal: bool = False):
        super().__init__()
        in_ch = 3 * LQ_PATCH * LQ_PATCH
        self.causal = causal
        self.conv1 = nn.Conv3d(in_ch, hidden1, (4, 3, 3), stride=(2, 1, 1))
        self.norm1 = ChannelRMSNorm(hidden1)
        self.conv2 = nn.Conv3d(hidden1, hidden2, (4, 3, 3), stride=(2, 1, 1))
        self.norm2 = ChannelRMSNorm(hidden2)
        self.linear = nn.Linear(hidden2, out_dim)

    @staticmethod
    def pixel_unshuffle(x):
        """[N, 3, T, H, W] -> [N, 3*16*16, T, H/16, W/16], channels ordered (c, dy, dx)."""
        n, c, t, hh, ww = x.shape
        x = x.reshape(n, c, t, hh // LQ_PATCH, LQ_PATCH, ww // LQ_PATCH, LQ_PATCH)
        return x.permute(0, 1, 4, 6, 2, 3, 5).reshape(n, c * LQ_PATCH * LQ_PATCH, t, hh // LQ_PATCH, ww // LQ_PATCH)

    @staticmethod
    def conv(conv, x, cache):
        # Two cached frames in front, replicate padding of the spatial borders.
        x = torch.cat([cache, x], dim=2)
        return conv(F.pad(x, (1, 1, 1, 1, 0, 0), mode="replicate"))

    def forward(self, lq_frames, conv1_cache=None, conv2_cache=None):
        _, c, t, hh, ww = lq_frames.shape
        n = t // 4
        if self.causal:
            # Clips one after another, every convolution sees the last two frames of its previous input.
            outs = []
            x_all = self.pixel_unshuffle(lq_frames)
            for i in range(n):
                x = x_all[:, :, 4 * i : 4 * i + 4]
                y = F.silu(self.norm1(self.conv(self.conv1, x, conv1_cache)))
                conv1_cache = x[:, :, -2:]
                z = F.silu(self.norm2(self.conv(self.conv2, y, conv2_cache)))
                conv2_cache = y[:, :, -2:]
                outs.append(z)
            z = torch.cat(outs, dim=2).transpose(1, 2).flatten(0, 1).unsqueeze(2)
        else:
            clips = lq_frames.reshape(1, c, n, 4, hh, ww).transpose(1, 2).reshape(n, c, 4, hh, ww)
            x = self.pixel_unshuffle(clips)
            y = F.silu(self.norm1(self.conv(self.conv1, x, x[:, :, -2:])))
            z = F.silu(self.norm2(self.conv(self.conv2, y, y[:, :, -2:])))
        # z: [n, hidden2, 1, h, w] -> [1, n, h, w, out_dim]
        out = self.linear(z[:, :, 0].permute(0, 2, 3, 1)).unsqueeze(0)
        if self.causal:
            return out, conv1_cache, conv2_cache
        return out

    @classmethod
    def from_checkpoint(cls, path: Path, causal: bool = False):
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            out_dim=state["linear_layers.0.weight"].shape[0],
            hidden1=state["conv1.weight"].shape[0],
            hidden2=state["conv2.weight"].shape[0],
            causal=causal,
        )
        renamed = {key.replace("linear_layers.0.", "linear."): value.float() for key, value in state.items()}
        model.load_state_dict(renamed, strict=True)
        return model.eval()


# ---------------------------------------------------------------------------------------------------------------------
# Tiny conditional decoder
# ---------------------------------------------------------------------------------------------------------------------
class MemBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_in * 2, n_out, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_out, n_out, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_out, n_out, 3, padding=1),
        )

    def forward(self, x, past):
        return F.relu(self.conv(torch.cat([x, past], 1)) + x)


class TCDecoder(nn.Module):
    """The tiny conditional decoder of FlashVSR (a widened, decoder-only TAEHV with the LQ frames as condition).

    Runs a chunk of T latent frames at once, time on the batch axis. Every MemBlock sees the input of the previous
    frame at its level: inside the chunk that is the previous batch row, for the first row it is the memory input
    (zeros at the start of a video). The memories of the next chunk are the inputs of the last frame of every MemBlock.

    Inputs: ``latents`` [T, 16, h, w], ``cond`` [T, 3*4*8*8, h, w] (4 LQ frames per latent frame, pixel-unshuffled by
    8 x 8), and the memories. Output: ``frames`` [4T, 3, 8h, 8w] in about [0, 1], then the new memories.
    """

    # Layer indices of the upstream nn.Sequential (with the identity deepening convolutions), used for the weights.
    STAGES = ((5, 9, 10, 1), (11, 15, 16, 2), (17, 21, 22, 2))

    def __init__(self, channels=(512, 256, 128, 128), latent_channels: int = 16, cond_channels: int = 768):
        super().__init__()
        n_f = channels
        self.conv_in = nn.Conv2d(latent_channels + cond_channels, n_f[0], 3, padding=1)
        self.conv_in_deep = nn.Conv2d(n_f[0], n_f[0], 3, padding=1, bias=False)
        self.mem_blocks = nn.ModuleList()
        self.t_grow = nn.ModuleList()
        self.stage_out = nn.ModuleList()
        self.t_strides = []
        for stage, (_, _, _, stride) in enumerate(self.STAGES):
            self.mem_blocks.extend(MemBlock(n_f[stage], n_f[stage]) for _ in range(3))
            self.t_grow.append(nn.Conv2d(n_f[stage], n_f[stage] * stride, 1, bias=False))
            self.stage_out.append(nn.Conv2d(n_f[stage], n_f[stage + 1], 3, padding=1, bias=False))
            self.t_strides.append(stride)
        self.conv_out_deep = nn.Conv2d(n_f[3], n_f[3], 3, padding=1, bias=False)
        self.conv_out = nn.Conv2d(n_f[3], 3, 3, padding=1)

    def memory_shapes(self, h: int, w: int):
        shapes = []
        for stage in range(3):
            shapes += [(1, self.mem_blocks[3 * stage].conv[0].in_channels // 2, h << stage, w << stage)] * 3
        return shapes

    def forward(self, latents, cond, memories):
        x = torch.cat([cond, latents], dim=1)
        x = torch.tanh(x / 3) * 3
        x = F.relu(self.conv_in(x))
        x = F.relu(self.conv_in_deep(x))
        new_memories = []
        for stage in range(3):
            for i in range(3):
                blk = self.mem_blocks[3 * stage + i]
                # The previous frame of every row (a T-1 sized slice would make torch.export specialize T).
                past = torch.cat([memories[3 * stage + i], x], dim=0)[: x.shape[0]]
                new_memories.append(x[-1:])
                x = blk(x, past)
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            stride = self.t_strides[stage]
            x = self.t_grow[stage](x)
            t, c, hh, ww = x.shape
            x = x.reshape(t * stride, c // stride, hh, ww)
            x = self.stage_out[stage](x)
        x = F.relu(x)
        x = F.relu(self.conv_out_deep(x))
        return self.conv_out(x), new_memories

    @classmethod
    def from_checkpoint(cls, path: Path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        channels = [state[f"decoder.{mem}.conv.0.weight"].shape[0] for mem, _, _, _ in cls.STAGES]
        channels.append(state["decoder.22.weight"].shape[0])
        model = cls(channels=tuple(channels), latent_channels=16, cond_channels=state["decoder.1.weight"].shape[1] - 16)
        renamed = {
            "conv_in.weight": state["decoder.1.weight"],
            "conv_in.bias": state["decoder.1.bias"],
            "conv_in_deep.weight": state["decoder.3.weight"],
            "conv_out_deep.weight": state["decoder.24.weight"],
            "conv_out.weight": state["decoder.26.weight"],
            "conv_out.bias": state["decoder.26.bias"],
        }
        for stage, (mem, grow, out, _) in enumerate(cls.STAGES):
            for i in range(3):
                for conv in (0, 2, 4):
                    for p in ("weight", "bias"):
                        renamed[f"mem_blocks.{3 * stage + i}.conv.{conv}.{p}"] = state[
                            f"decoder.{mem + i}.conv.{conv}.{p}"
                        ]
            renamed[f"t_grow.{stage}.weight"] = state[f"decoder.{grow}.conv.weight"]
            renamed[f"stage_out.{stage}.weight"] = state[f"decoder.{out}.weight"]
        model.load_state_dict({k: v.float() for k, v in renamed.items()}, strict=True)
        return model.eval()


def pixel_unshuffle_cond(frames: torch.Tensor) -> torch.Tensor:
    """[1, 3, 4T, H, W] LQ frames -> [T, 3*4*8*8, H/8, W/8] decoder condition, channels ordered (c, t, dy, dx)."""
    _, c, t, hh, ww = frames.shape
    x = frames.reshape(c, t // 4, 4, hh // 8, 8, ww // 8, 8)
    return x.permute(1, 0, 2, 4, 6, 3, 5).reshape(t // 4, c * 4 * 64, hh // 8, ww // 8)
