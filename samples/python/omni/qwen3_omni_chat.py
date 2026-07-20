#!/usr/bin/env python3
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Omni multimodal chat sample.

Preview: the Qwen3-Omni API (OmniPipeline and related types) is a preview feature
and is subject to change in future releases.

Demonstrates text + image + audio -> text + speech output using the ChatHistory API.

Usage:
    python qwen3_omni_chat.py <MODEL_DIR> <IMAGE_FILE_OR_DIR> [--audio AUDIO_WAV]
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import openvino_genai
import soundfile as sf
from openvino import Tensor
from PIL import Image


def streamer(subword: str) -> openvino_genai.StreamingStatus:
    """Stream text tokens to stdout."""
    print(subword, end="", flush=True)
    return openvino_genai.StreamingStatus.RUNNING


def read_image(path: str) -> Tensor:
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic)
    return Tensor(image_data)


def read_images(path: str) -> list[Tensor]:
    entry = Path(path)
    if entry.is_dir():
        return [read_image(str(file)) for file in sorted(entry.iterdir())]
    return [read_image(path)]


def load_audio(audio_path: str, target_sr: int = 16000) -> Tensor:
    """Load audio from WAV file and convert to float32 mono tensor at target_sr."""
    audio_data, sample_rate = sf.read(audio_path, dtype="float32")

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    if sample_rate != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)

    return Tensor(audio_data.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-Omni multimodal chat")
    parser.add_argument("model_dir", help="Path to the OpenVINO model directory")
    parser.add_argument("image_dir", help="Image file or directory with images")
    parser.add_argument("--audio", help="Path to input audio WAV file (optional)")
    args = parser.parse_args()

    rgbs = read_images(args.image_dir)

    pipe = openvino_genai.OmniPipeline(args.model_dir, "CPU")

    # Two configs: text_config drives the thinker text decode; talker_speech_config drives the
    # talker + speech output. Speech output is hardcoded on here to show the multimodal path.
    # Set talker_speech_config.return_audio = False to get text-only responses.
    text_config = openvino_genai.GenerationConfig()
    text_config.max_new_tokens = 256

    talker_speech_config = openvino_genai.OmniTalkerSpeechConfig(args.model_dir)
    talker_speech_config.return_audio = True
    # Leaving speaker empty selects the model's default voice. Available voices vary by checkpoint
    # (e.g. MoE exposes "Ethan", "Chelsie", "Aiden", "Cherry"); the full list is in
    # talker_config.speaker_id of the model's config.json.

    videos = []
    audios = [load_audio(args.audio)] if args.audio else []

    history = openvino_genai.ChatHistory()
    prompt = input("question:\n")
    history.append({"role": "user", "content": prompt})
    decoded_results = pipe.generate(
        history,
        images=rgbs,
        videos=videos,
        audios=audios,
        text_config=text_config,
        talker_speech_config=talker_speech_config,
        streamer=streamer,
    )
    history.append({"role": "assistant", "content": decoded_results.texts[0]})

    if decoded_results.speech_result.waveforms:
        print(f"\n[Speech output: {decoded_results.speech_result.waveforms[0].get_size()} samples at 24kHz]")

    while True:
        try:
            prompt = input("\n----------\nquestion:\n")
        except EOFError:
            break

        history.append({"role": "user", "content": prompt})
        # New images can be passed at each turn; here we rely on the info from turn 1.
        images = []
        decoded_results = pipe.generate(
            history,
            images=images,
            videos=videos,
            audios=audios,
            text_config=text_config,
            talker_speech_config=talker_speech_config,
            streamer=streamer,
        )
        history.append({"role": "assistant", "content": decoded_results.texts[0]})

        if decoded_results.speech_result.waveforms:
            print(f"\n[Speech output: {decoded_results.speech_result.waveforms[0].get_size()} samples at 24kHz]")


if __name__ == "__main__":
    main()
