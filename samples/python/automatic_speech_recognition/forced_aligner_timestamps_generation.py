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
    parser.add_argument("asr_model_dir", help="Path to the ASR model directory")
    parser.add_argument("forced_aligner_model_dir", help="Path to the Qwen3 forced-aligner model directory")
    parser.add_argument("wav_file_path", help="Path to the WAV file")
    parser.add_argument("device", nargs="?", default="CPU", help="Device to run the model on (default: CPU)")
    parser.add_argument("--language", help="Language for alignment (e.g. English or Chinese)")
    args = parser.parse_args()

    pipe = openvino_genai.ASRPipeline(args.asr_model_dir, args.device)
    aligner = openvino_genai.ASRForcedAligner(args.forced_aligner_model_dir, args.device)

    # ASR pipelines expect normalized 16 kHz audio.
    raw_speech = read_wav(args.wav_file_path)

    # Transcribe audio using the ASR model.
    result = pipe.generate(raw_speech)
    transcript = result.texts[0]

    print(f"Transcription: {transcript}")

    # Use the detected language when available, otherwise use the specified language.
    language = args.language or (result.languages[0] if result.languages else None)

    if not language:
        parser.error("Specify --language when the ASR model does not report the language")

    # Align the transcription with the audio to get word timestamps.
    words = aligner.align(raw_speech, transcript, language=language)

    print()
    print("Word timestamps:")
    for word in words:
        print(f"[{word.start_ts:.2f}, {word.end_ts:.2f}]: {word.text}")


if "__main__" == __name__:
    main()