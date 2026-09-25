#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
import librosa


def read_wav(filepath):
    raw_speech, _ = librosa.load(filepath, sr=16000)
    return raw_speech.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to the forced-aligner model directory")
    parser.add_argument("wav_file_path", help="Path to the WAV file")
    parser.add_argument("transcript", help="Transcript to align to the audio")
    parser.add_argument("device", nargs="?", default="CPU", help="Device to run the model on (default: CPU)")
    parser.add_argument("--language", default="english", help="Transcript language (default: english)")
    args = parser.parse_args()

    aligner = openvino_genai.ASRForcedAligner(args.model_dir, args.device)

    raw_speech = read_wav(args.wav_file_path)
    words = aligner.align(raw_speech, args.transcript, language=args.language)

    for word in words:
        print(f"[{word.start_ts:.2f}, {word.end_ts:.2f}]: {word.text}")


if "__main__" == __name__:
    main()
