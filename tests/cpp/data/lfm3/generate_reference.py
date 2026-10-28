# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import __future__

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers.audio_utils import mel_filter_bank
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor

ROOT = Path(__file__).parent
SOURCES = {
    "feature_extraction_lfm3_omni.py": "ce2458cbce4ad970f92032ed94e3d18d30b04f9c",
    "modeling_liquid_codec.py": "ce2458cbce4ad970f92032ed94e3d18d30b04f9c",
}


def publisher_class(filename, classname):
    tree = ast.parse((ROOT / filename).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {
        "np": np, "torch": torch, "nn": nn, "F": F,
        "SequenceFeatureExtractor": SequenceFeatureExtractor, "mel_filter_bank": mel_filter_bank,
    }
    exec(compile(module, filename, "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[classname]


def save(name, value):
    value.contiguous().numpy().astype("float32").tofile(ROOT / name)


extractor = publisher_class("feature_extraction_lfm3_omni.py", "Lfm3OmniFeatureExtractor")()
istft = publisher_class("modeling_liquid_codec.py", "LiquidCodecISTFT")(
    SimpleNamespace(stft_n_fft=1280, stft_hop_length=320, stft_win_length=1280)
)
manifest = {"torch": torch.__version__, "sources": {}, "audio": [], "istft": []}
for filename, revision in SOURCES.items():
    manifest["sources"][filename] = {
        "revision": revision, "sha256": hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(),
        "url": f"https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/{revision}/{filename}",
    }
for samples in (1, 400, 1281, 16000, 48001):
    time = torch.arange(samples, dtype=torch.float32)
    waveform = .13 * torch.sin(time * .021) + .031 * torch.cos(time * .137)
    padded = F.pad(waveform, (0, extractor.pad_length(samples) - samples))
    features = extractor._log_mel(padded).T[None]
    save(f"waveform-{samples}.bin", waveform)
    save(f"features-{samples}.bin", features)
    manifest["audio"].append({"samples": samples, "feature_shape": list(features.shape)})
for frames in (1, 6, 18, 37):
    generator = torch.Generator().manual_seed(77 + frames)
    log_magnitude = -4. + .4 * torch.randn((1, 641, frames), generator=generator)
    angle = 3. * torch.randn((1, 641, frames), generator=generator)
    spectral = torch.cat((log_magnitude, angle), dim=1).mT.contiguous()
    audio, envelope = istft(torch.polar(log_magnitude.exp(), angle))
    waveform = (audio[:, 480:-480] / envelope[480:-480]).float()
    save(f"spectral-{frames}.bin", spectral)
    save(f"istft-{frames}.bin", waveform)
    manifest["istft"].append({"frames": frames, "samples": waveform.numel()})
save("istft-window.bin", istft.window)
(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"audio": len(manifest["audio"]), "istft": len(manifest["istft"]), "torch": torch.__version__}))
