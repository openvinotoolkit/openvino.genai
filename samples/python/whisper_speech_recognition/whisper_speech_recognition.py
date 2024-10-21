#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
import librosa


def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=16000)
    return raw_speech.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("wav_file_path")
    args = parser.parse_args()

    raw_speech = read_wav(args.wav_file_path)

    config = openvino_genai.WhisperGenerationConfig(
        args.model_dir + "/generation_config.json"
    )
    config.max_new_tokens = 100
    # 'task' and 'language' parameters are supported for multilingual models only
    config.language = "<|en|>"
    config.task = "transcribe"
    config.return_timestamps = True

    pipe = openvino_genai.WhisperPipeline(args.model_dir)

    def streamer(word: str) -> bool:
        print(word, end="")
        return False

    result = pipe.generate(raw_speech, config, streamer)

    print()

    for chunk in result.chunks:
        print(f"timestamps: [{chunk.start_ts}, {chunk.end_ts}] text: {chunk.text}")


if "__main__" == __name__:
    main()
