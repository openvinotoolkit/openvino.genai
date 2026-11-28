# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch definition of the FunASR FSMN-VAD encoder and the loading of its official weights.

The architecture is adapted from funasr/models/fsmn_vad_streaming/encoder.py of FunASR 1.4.16
(https://github.com/modelscope/FunASR), the FSMN encoder with the explicit layer caches of its FSMNExport wrapper,
MIT License, Copyright (c) 2025 FunASR. The license text is in third-party-programs.txt.
"""

from pathlib import Path

import torch
from torch import nn

ENCODER_CONF = {
    "input_dim": 400,
    "input_affine_dim": 140,
    "fsmn_layers": 4,
    "linear_dim": 250,
    "proj_dim": 128,
    "lorder": 20,
    "rorder": 0,
    "lstride": 1,
    "rstride": 0,
    "output_affine_dim": 140,
    "output_dim": 248,
}
CACHE_SHAPE = (1, ENCODER_CONF["proj_dim"], ENCODER_CONF["lorder"] - 1, 1)

_CHECKPOINT_PREFIX = "encoder."


class _Linear(nn.Module):
    """FunASR's AffineTransform / LinearTransform, kept for the parameter names of the checkpoint."""

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _FSMNBlock(nn.Module):
    def __init__(self, dim: int, lorder: int):
        super().__init__()
        self.cache_frames = lorder - 1
        self.conv_left = nn.Conv2d(dim, dim, [lorder, 1], groups=dim, bias=False)

    def forward(self, x: torch.Tensor, cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_per = x.unsqueeze(1).permute(0, 3, 2, 1)
        y_left = torch.cat((cache, x_per), dim=2)
        out = x_per + self.conv_left(y_left)
        return out.permute(0, 3, 2, 1).squeeze(1), y_left[:, :, -self.cache_frames :, :]


class _BasicBlock(nn.Module):
    def __init__(self, linear_dim: int, proj_dim: int, lorder: int):
        super().__init__()
        self.linear = _Linear(linear_dim, proj_dim, bias=False)
        self.fsmn_block = _FSMNBlock(proj_dim, lorder)
        self.affine = _Linear(proj_dim, linear_dim)

    def forward(self, x: torch.Tensor, cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, out_cache = self.fsmn_block(self.linear(x), cache)
        return torch.relu(self.affine(x)), out_cache


class FSMNEncoder(nn.Module):
    """``speech`` [1, T, 400] (LFR features after CMVN) and the four layer caches [1, 128, 19, 1] (zeros at the
    start of a recording) give the frame posteriors ``logits`` [1, T, 248] and the caches for the next frames."""

    def __init__(self):
        super().__init__()
        conf = ENCODER_CONF
        self.in_linear1 = _Linear(conf["input_dim"], conf["input_affine_dim"])
        self.in_linear2 = _Linear(conf["input_affine_dim"], conf["linear_dim"])
        self.fsmn = nn.ModuleList(
            _BasicBlock(conf["linear_dim"], conf["proj_dim"], conf["lorder"]) for _ in range(conf["fsmn_layers"])
        )
        self.out_linear1 = _Linear(conf["linear_dim"], conf["output_affine_dim"])
        self.out_linear2 = _Linear(conf["output_affine_dim"], conf["output_dim"])

    def forward(
        self,
        speech: torch.Tensor,
        in_cache0: torch.Tensor,
        in_cache1: torch.Tensor,
        in_cache2: torch.Tensor,
        in_cache3: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        x = torch.relu(self.in_linear2(self.in_linear1(speech)))
        out_caches = []
        for block, cache in zip(self.fsmn, (in_cache0, in_cache1, in_cache2, in_cache3)):
            x, out_cache = block(x, cache)
            out_caches.append(out_cache)
        logits = torch.softmax(self.out_linear2(self.out_linear1(x)), dim=-1)
        return (logits, *out_caches)


def load_encoder(checkpoint: Path) -> FSMNEncoder:
    """Loads the encoder weights of the official FunASR ``model.pt``; every tensor must match the architecture."""
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise ValueError(f"{checkpoint} does not hold a state dict")
    encoder = FSMNEncoder()
    state_dict = {key.removeprefix(_CHECKPOINT_PREFIX): value for key, value in state_dict.items()}
    encoder.load_state_dict(state_dict, strict=True)
    return encoder.eval()
