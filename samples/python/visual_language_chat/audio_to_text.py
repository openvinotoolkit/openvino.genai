#!/usr/bin/env python3
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Simple audio-to-text sample using Qwen3-Omni.

Processes an audio file and generates a text response.

Usage:
    python audio_to_text.py <MODEL_DIR> <AUDIO_WAV>
"""

import argparse

import librosa
import numpy as np
import openvino as ov
import openvino_genai as ov_genai
import soundfile as sf


def load_audio(audio_path: str, target_sr: int = 16000) -> ov.Tensor:
    """Load audio from WAV file and convert to float32 mono tensor at target_sr."""
    audio_data, sample_rate = sf.read(audio_path, dtype="float32")

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    if sample_rate != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)

    return ov.Tensor(audio_data.astype(np.float32))


def streamer(subword: str) -> ov_genai.StreamingStatus:
    print(subword, end="", flush=True)
    return ov_genai.StreamingStatus.RUNNING


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio-to-text with Qwen3-Omni")
    parser.add_argument("model_dir", help="Path to OpenVINO model directory")
    parser.add_argument("audio_path", help="Path to input audio WAV file")
    args = parser.parse_args()

    # Prompt is hardcoded to keep the sample focused on the audio path.
    prompt = "Describe what you hear."

    pipe = ov_genai.VLMPipeline(args.model_dir, "CPU")

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 256

    audio_tensor = load_audio(args.audio_path)

    pipe.generate(
        prompt,
        generation_config=config,
        audios=[audio_tensor],
        streamer=streamer,
    )
    print()


if __name__ == "__main__":
    main()
