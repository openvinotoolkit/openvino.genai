#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from concurrent.futures import ThreadPoolExecutor

import librosa
import openvino_genai as ov_genai


SAMPLE_RATE = 16000
FRAME_SAMPLES = SAMPLE_RATE // 10


def main():
    parser = argparse.ArgumentParser(description="Stream audio to the Voxtral Realtime ASR pipeline.")
    parser.add_argument("model_dir", help="Path to the exported Voxtral Realtime model")
    parser.add_argument("audio_file", help="Path to an audio file")
    parser.add_argument("device", nargs="?", default="CPU", help="Inference device (default: CPU)")
    args = parser.parse_args()

    audio, _ = librosa.load(args.audio_file, sr=SAMPLE_RATE, mono=True)
    pipe = ov_genai.ASRPipeline(args.model_dir, args.device)
    stream = pipe.create_stream()

    def push_audio():
        for start in range(0, len(audio), FRAME_SAMPLES):
            stream.push_audio(audio[start : start + FRAME_SAMPLES].tolist())
        stream.end_audio()

    with ThreadPoolExecutor(max_workers=1) as executor:
        producer = executor.submit(push_audio)
        result = stream.generate(lambda delta: print(delta, end="", flush=True))
        producer.result()

    print("\nFinal:", result.texts[0])


if __name__ == "__main__":
    main()
