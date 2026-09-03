#!/usr/bin/env python3
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Omni audio placement sample.

Preview: the Qwen3-Omni API (OmniPipeline and related types) is a preview feature
and is subject to change in future releases.

Shows where an audio lands in the prompt. Each audio you pass gets an index, and
`<ov_genai_audio_N>` puts audio N at that exact spot. Leave the tags out and the audio
is prepended instead.

Usage:
    python audio_placement.py <MODEL_DIR> <AUDIO_WAV> [<AUDIO_WAV> ...]
"""

import argparse

import librosa
import numpy as np
import openvino_genai
import soundfile as sf
from openvino import Tensor


def load_audio(audio_path: str, target_sr: int = 16000) -> Tensor:
    """Load a WAV file as a mono float32 tensor at target_sr, the layout Qwen3-Omni expects."""
    audio_data, sample_rate = sf.read(audio_path, dtype="float32")

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    if sample_rate != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)

    return Tensor(audio_data.astype(np.float32))


def run(pipe: openvino_genai.OmniPipeline, label: str, prompt: str, audios: list[Tensor]) -> None:
    text_config = openvino_genai.GenerationConfig()
    text_config.max_new_tokens = 60
    text_config.do_sample = False

    # Text only. Speech output has its own sample; here the point is prompt layout.
    talker_config = openvino_genai.OmniTalkerSpeechConfig()
    talker_config.return_audio = False

    print(f"\n--- {label} ---")
    print(f"prompt: {prompt!r}")
    results = pipe.generate(
        prompt,
        audios=audios,
        text_config=text_config,
        talker_speech_config=talker_config,
    )
    print(f"tokens in prompt: {results.perf_metrics.get_num_input_tokens()}")
    print(f"answer: {results.texts[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Where audio lands in a Qwen3-Omni prompt")
    parser.add_argument("model_dir", help="Path to the OpenVINO model directory")
    parser.add_argument("audio", nargs="+", metavar="WAV", help="One or more input audio WAV files")
    args = parser.parse_args()

    audios = [load_audio(path) for path in args.audio]
    pipe = openvino_genai.OmniPipeline(args.model_dir, "CPU")

    # No tag: audio goes in front. This is the default and the layout the model was trained on.
    run(pipe, "no tag (prepended)", "Describe what you hear.", audios[:1])

    # A tag puts the audio where you write it. Same token count as above, different position.
    run(pipe, "tagged at the front", "<ov_genai_audio_0> Describe what you hear.", audios[:1])

    # Text on both sides. Supported, but off-distribution, so answers may be weaker.
    run(pipe, "interleaved", "Listen closely: <ov_genai_audio_0> now describe it.", audios[:1])

    if len(audios) >= 2:
        # The index picks the audio, so any order works, and repeats are fine.
        run(
            pipe,
            "two audios, in order",
            "First <ov_genai_audio_0>, then <ov_genai_audio_1>. What changed?",
            audios[:2],
        )
        run(
            pipe,
            "two audios, order swapped",
            "First <ov_genai_audio_1>, then <ov_genai_audio_0>. What changed?",
            audios[:2],
        )
    else:
        print("\n(pass a second WAV file to see two audios addressed by index)")


if __name__ == "__main__":
    main()
