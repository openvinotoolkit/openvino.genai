#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
from scipy.io import wavfile
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("wav_file_path")
    args = parser.parse_args()

    model_dir = args.model_dir

    samplerate, raw_speech = wavfile.read(args.wav_file_path)

    # normalize to [-1 1] range
    raw_speech = (raw_speech / np.iinfo(raw_speech.dtype).max).tolist()

    pipe = openvino_genai.WhisperSpeechRecognitionPipeline(model_dir)

    def streamer(word: str) -> bool:
        print(word, end="")
        return False

    pipe.generate(raw_speech, streamer=streamer)


if "__main__" == __name__:
    main()
