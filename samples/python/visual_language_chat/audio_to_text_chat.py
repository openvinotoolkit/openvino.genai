#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import librosa
import openvino_genai
from openvino import Tensor


def streamer(subword: str) -> None:
    print(subword, end="", flush=True)


def read_wav(path: str) -> Tensor:
    audio, _ = librosa.load(path, sr=16000, mono=True)
    return Tensor(audio)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("audio_file", help="Path to the WAV file")
    parser.add_argument("device", nargs="?", default="CPU", help="Device to run the model on (default: CPU)")
    args = parser.parse_args()

    audio = read_wav(args.audio_file)

    properties = {}
    if args.device == "GPU":
        properties["CACHE_DIR"] = "vlm_cache"
    pipe = openvino_genai.VLMPipeline(args.model_dir, args.device, **properties)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    history = openvino_genai.ChatHistory()
    prompt = input("question:\n")
    history.append({"role": "user", "content": prompt})
    decoded_results = pipe.generate(history, audios=[audio], generation_config=config, streamer=streamer)
    history.append({"role": "assistant", "content": decoded_results.texts[0]})

    while True:
        try:
            prompt = input("\n----------\nquestion:\n")
        except EOFError:
            break

        history.append({"role": "user", "content": prompt})
        decoded_results = pipe.generate(history, generation_config=config, streamer=streamer)
        history.append({"role": "assistant", "content": decoded_results.texts[0]})


if __name__ == "__main__":
    main()
