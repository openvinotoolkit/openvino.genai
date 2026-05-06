#!/usr/bin/env python3
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Simple audio-to-text sample using Qwen3-Omni.

Processes an audio file and generates a text response.

Usage:
    python audio_to_text.py model_dir audio.wav "Describe what you hear in this audio"
"""

import argparse
import sys

import numpy as np
import openvino as ov
import openvino_genai as ov_genai


def load_audio(audio_path: str, target_sr: int = 16000) -> ov.Tensor:
    """Load audio from WAV file and convert to float32 tensor."""
    try:
        import soundfile as sf
    except ImportError:
        print("Error: soundfile package required. Install with: pip install soundfile")
        sys.exit(1)

    audio_data, sample_rate = sf.read(audio_path, dtype="float32")

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    if sample_rate != target_sr:
        try:
            import librosa

            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
        except ImportError:
            print(f"Warning: Audio is {sample_rate}Hz, expected {target_sr}Hz. Install librosa for resampling.")

    return ov.Tensor(audio_data.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio-to-text with Qwen3-Omni")
    parser.add_argument("model_dir", type=str, help="Path to OpenVINO model directory")
    parser.add_argument("audio_path", type=str, help="Path to input audio WAV file")
    parser.add_argument("prompt", type=str, nargs="?", default="Describe what you hear.", help="Text prompt")
    parser.add_argument("--device", type=str, default="CPU", help="Inference device")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum tokens to generate")
    args = parser.parse_args()

    from pathlib import Path

    if not Path(args.model_dir).is_dir():
        print(f"Error: model directory not found: {args.model_dir}")
        sys.exit(1)
    if not Path(args.audio_path).is_file():
        print(f"Error: audio file not found: {args.audio_path}")
        sys.exit(1)

    print(f"Loading model from: {args.model_dir}")
    pipe = ov_genai.VLMPipeline(args.model_dir, args.device)

    print(f"Loading audio: {args.audio_path}")
    audio_tensor = load_audio(args.audio_path)
    print(f"Audio: {audio_tensor.size} samples ({audio_tensor.size / 16000:.1f}s at 16kHz)")

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens

    print(f"\nPrompt: {args.prompt}")
    print("Response: ", end="", flush=True)

    result = pipe.generate(
        args.prompt,
        generation_config=config,
        audios=[audio_tensor],
        streamer=lambda text: print(text, end="", flush=True) or False,
    )
    print()


if __name__ == "__main__":
    main()
