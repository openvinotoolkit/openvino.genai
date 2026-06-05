#!/usr/bin/env python3
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Omni multimodal chat sample.

Demonstrates text + image -> text + speech output using the ChatHistory API.

Usage:
    python qwen3_omni_chat.py <MODEL_DIR> <IMAGE_FILE_OR_DIR>
"""

import argparse
from pathlib import Path

import numpy as np
import openvino_genai
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-Omni multimodal chat")
    parser.add_argument("model_dir", help="Path to the OpenVINO model directory")
    parser.add_argument("image_dir", help="Image file or directory with images")
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

    history = openvino_genai.ChatHistory()
    prompt = input("question:\n")
    history.append({"role": "user", "content": prompt})
    decoded_results = pipe.generate(
        history,
        images=rgbs,
        videos=[],
        audios=[],
        text_config=text_config,
        talker_speech_config=talker_speech_config,
        streamer=streamer,
    )
    history.append({"role": "assistant", "content": decoded_results.texts[0]})

    if decoded_results.speech_outputs:
        print(f"\n[Speech output: {decoded_results.speech_outputs[0].get_size()} samples at 24kHz]")

    while True:
        try:
            prompt = input("\n----------\nquestion:\n")
        except EOFError:
            break

        history.append({"role": "user", "content": prompt})
        # New images can be passed at each turn; here we only pass them on turn 1.
        decoded_results = pipe.generate(
            history,
            images=[],
            videos=[],
            audios=[],
            text_config=text_config,
            talker_speech_config=talker_speech_config,
            streamer=streamer,
        )
        history.append({"role": "assistant", "content": decoded_results.texts[0]})

        if decoded_results.speech_outputs:
            print(f"\n[Speech output: {decoded_results.speech_outputs[0].get_size()} samples at 24kHz]")


if __name__ == "__main__":
    main()
