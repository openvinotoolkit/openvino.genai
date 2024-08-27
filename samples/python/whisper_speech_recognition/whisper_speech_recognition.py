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

    device = "CPU"  # GPU can be used as well
    pipe = openvino_genai.WhisperSpeechRecognitionPipeline(args.model_dir, device)

    samplerate, data = wavfile.read(args.wav_file_path)

    # normalize to [-1 1] range
    data = data / np.iinfo(data.dtype).max

    result = pipe.generate(data)
    print(result)


if "__main__" == __name__:
    main()
