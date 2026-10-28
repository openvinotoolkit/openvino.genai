# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generate preprocessing and ISTFT expectations using pinned LiquidAI sources."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

REVISION = "19e65845923a7f136442c95137884ec61eb386aa"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liquid-audio", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.liquid_audio), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Expected LiquidAI revision {REVISION}, got {revision}")
    source = args.liquid_audio / "src/liquid_audio/model/conformer/processor.py"
    config = json.loads(args.config.read_text())["preprocessor"]
    processor = load_module(source, "reference_processor").AudioToMelSpectrogramPreprocessor(**config).eval()
    output = Path(__file__).parent
    report = {"source_revision": revision, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "config": config, "scenarios": []}
    for count in [160, 1281, 16000]:
        t = torch.arange(count, dtype=torch.float32)
        waveform = (0.3 * torch.sin(t * 0.13) + 0.07 * torch.cos(t * 0.31)).unsqueeze(0)
        features, length = processor(waveform, torch.tensor([count]))
        waveform.numpy().tofile(output / f"waveform-{count}.bin")
        features.numpy().tofile(output / f"features-{count}.bin")
        report["scenarios"].append({"samples": count, "shape": list(features.shape), "length": length.tolist(),
            "features_sha256": hashlib.sha256((output / f"features-{count}.bin").read_bytes()).hexdigest()})
    detokenizer_source = args.liquid_audio / "src/liquid_audio/detokenizer.py"
    istft = load_module(detokenizer_source, "reference_detokenizer").ISTFT(1280, 320, 1280)
    report["istft_source_sha256"] = hashlib.sha256(detokenizer_source.read_bytes()).hexdigest()
    report["istft_scenarios"] = []
    torch.hann_window(1280).numpy().tofile(output / "istft-window.bin")
    for frames in [1, 18, 37]:
        t = torch.arange(frames * 1282, dtype=torch.float32).reshape(1, frames, 1282)
        spectral = torch.cat([-2.0 + 0.3 * torch.sin(t[:, :, :641] * 0.07),
                              0.8 * torch.cos(t[:, :, 641:] * 0.11)], dim=-1)
        magnitude, angle = spectral.mT.chunk(2, dim=1)
        waveform = istft(torch.polar(magnitude.exp(), angle))
        spectral.numpy().tofile(output / f"spectral-{frames}.bin")
        waveform.numpy().tofile(output / f"istft-{frames}.bin")
        report["istft_scenarios"].append({"frames": frames, "samples": waveform.numel()})
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
