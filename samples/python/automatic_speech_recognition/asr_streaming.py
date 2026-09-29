#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import time
from typing import Any

import librosa
import numpy as np

try:
    import openvino_genai
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "OpenVINO GenAI Python package is not installed in the active environment. "
        "Build or install the repo in this interpreter before running the streaming ASR sample."
    ) from exc

if not hasattr(openvino_genai, "ASRStreamingConfig"):
    raise RuntimeError(
        "The active openvino_genai package does not include the streaming ASR API. "
        "Rebuild/install the repo from this checkout into the current Python environment."
    )


SAMPLE_RATE = 16000


def read_wav(path: str) -> np.ndarray:
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return np.asarray(audio, dtype=np.float32)


def print_result(result: Any, inline: bool) -> None:
    if result is None:
        return
    if inline:
        print(result.new_committed_text, end="", flush=True)
    else:
        print(f"[partial] ({result.language}) +{result.new_committed_text} [{result.partial_text}]")


def run_wav_streaming(pipe: Any, wav_path: str, cfg: Any,
                     inline: bool, block_ms: int = 250) -> Any:
    audio = read_wav(wav_path)
    session = pipe.create_streaming_session(cfg)

    block_size = max(1, int((block_ms / 1000.0) * SAMPLE_RATE))
    for start in range(0, len(audio), block_size):
        end = min(start + block_size, len(audio))
        print_result(session.push_chunk(audio[start:end].tolist()), inline)

    final = session.finish()
    print_result(final, inline)
    return final


def run_microphone_streaming(pipe: Any, cfg: Any,
                            inline: bool, duration_sec: float, block_ms: int = 250) -> Any:
    try:
        import pyaudio
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Microphone streaming requires the optional 'pyaudio' package. "
            "Install it with: pip install pyaudio"
        ) from exc

    session = pipe.create_streaming_session(cfg)

    block_size = max(1, int((block_ms / 1000.0) * SAMPLE_RATE))
    total_samples = int(duration_sec * SAMPLE_RATE)
    collected = 0

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                   channels=1,
                   rate=SAMPLE_RATE,
                   input=True,
                   frames_per_buffer=block_size)

    print("Microphone is live. Start speaking now.")
    try:
        while collected < total_samples:
            remaining = total_samples - collected
            samples_to_read = min(block_size, remaining)
            data = stream.read(samples_to_read, exception_on_overflow=False)
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            print_result(session.push_chunk(chunk.tolist()), inline)
            collected += len(chunk)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    final = session.finish()
    print_result(final, inline)
    return final


def build_streaming_config(args) -> Any:
    cfg = openvino_genai.ASRStreamingConfig()
    cfg.chunk_size_sec = args.chunk_size_sec
    cfg.warmup_chunks = args.warmup_chunks
    cfg.context_rollback_tokens = args.context_rollback_tokens
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Streaming ASR sample for OpenVINO GenAI")
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("--device", default="CPU", help="Device to run the model on (default: CPU)")
    parser.add_argument("--chunk-size-sec", type=float, default=2.0,
                        help="Audio duration per decode pass (default: %(default)s)")
    parser.add_argument("--warmup-chunks", type=int, default=2,
                        help="Number of startup chunks without prefix (default: %(default)s)")
    parser.add_argument("--context-rollback-tokens", type=int, default=5,
                        help="How many trailing tokens to rewind when reusing prior output as prefix (default: %(default)s)")
    parser.add_argument("--wav", type=str, help="Path to a WAV file. If omitted, microphone input is used.")
    parser.add_argument("--duration", type=float,
                        help="Microphone capture duration in seconds. Required when --wav is not set. (default: %(default)s)")
    parser.add_argument("--block-ms", type=int, default=250,
                        help="Audio chunk size in milliseconds for both WAV and microphone streaming (default: %(default)s)")
    parser.add_argument("--inline", action="store_true",
                        help="Print only newly committed text without newlines between chunks")
    args = parser.parse_args()

    if args.wav is None and args.duration is None:
        parser.error("Either --wav or --duration must be provided. Use --duration for microphone mode.")

    cfg = build_streaming_config(args)
    pipe = openvino_genai.ASRPipeline(args.model_dir, args.device)

    if args.wav is not None:
        final = run_wav_streaming(pipe, args.wav, cfg, inline=args.inline, block_ms=args.block_ms)
    else:
        final = run_microphone_streaming(pipe, cfg, inline=args.inline, duration_sec=args.duration, block_ms=args.block_ms)

    if args.inline:
        print()  # newline after the inline committed text stream
    print(f"\n[final] ({final.language}) {final.committed_text}")


if __name__ == "__main__":
    main()
