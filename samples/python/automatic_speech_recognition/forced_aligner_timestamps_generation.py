#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
import librosa


def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=16000)
    return raw_speech.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("asr_model_dir", help="Path to the Qwen3-ASR model directory")
    parser.add_argument("forced_aligner_model_dir", help="Path to the Qwen3 forced-aligner model directory")
    parser.add_argument("wav_file_path", help="Path to the WAV file")
    parser.add_argument("device", nargs="?", default="CPU", help="Device to run the model on (default: CPU)")
    args = parser.parse_args()

    aligner = openvino_genai.ASRForcedAligner(args.forced_aligner_model_dir, args.device)
    pipe = openvino_genai.ASRPipeline(args.asr_model_dir, args.device, forced_aligner=aligner)

    # ASR pipelines expect normalized 16 kHz audio.
    raw_speech = read_wav(args.wav_file_path)

    # Leave language unset to let Qwen3-ASR detect it from the audio.
    result = pipe.generate(raw_speech, word_timestamps=True)

    print(f"Detected language: {result.languages[0]}")
    print(f"Transcription: {result.texts[0]}")
    print()
    print("Word timestamps:")
    for word in result.words[0]:
        print(f"[{word.start_ts:.2f}, {word.end_ts:.2f}]: {word.text}")


if "__main__" == __name__:
    main()
