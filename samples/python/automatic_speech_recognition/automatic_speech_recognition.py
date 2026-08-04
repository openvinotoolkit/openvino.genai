#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
import librosa


def format_mean_std(pair, unit="ms"):
    return f"{pair.mean:.2f} +- {pair.std:.2f} {unit}"


def print_perf_metrics(result):
    perf = result.perf_metrics
    asr_raw = perf.asr_raw_metrics

    print("\n=== Performance Metrics ===")
    print(f"Load time: {perf.get_load_time():.2f} ms")
    print(f"Total generate duration: {format_mean_std(perf.get_generate_duration())}")
    print(f"Encoder inference: {format_mean_std(perf.get_encode_inference_duration())}")
    print(f"Decoder TTFT: {format_mean_std(perf.get_ttft())}")
    print(f"Decoder inference: {format_mean_std(perf.get_decode_inference_duration())}")
    print(f"TPOT: {format_mean_std(perf.get_tpot(), 'ms/token')}")
    print(f"Throughput: {format_mean_std(perf.get_throughput(), 'tokens/s')}")

    if asr_raw.encode_inference_durations:
        print(f"Encoder raw calls: {len(asr_raw.encode_inference_durations)}")
    if asr_raw.decode_inference_durations:
        print(f"Decoder raw steps: {len(asr_raw.decode_inference_durations)}")


def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=16000)
    return raw_speech.tolist()


def get_config_for_cache():
    config_cache = dict()
    return config_cache
    config_cache["CACHE_DIR"] = "asr_cache"
    return config_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("wav_file_path", help="Path to the WAV file")
    parser.add_argument("device", nargs="?", default="CPU", help="Device to run the model on (default: CPU)")
    args = parser.parse_args()

    ov_config = dict()
    if args.device == "NPU" or "GPU" in args.device:  # need to handle cases like "GPU", "GPU.0" and "GPU.1"
        # Cache compiled models on disk for GPU and NPU to save time on the
        # next run. It's not beneficial for CPU.
        ov_config = get_config_for_cache()

    # Word timestamps supported by Whisper models only
    # Must be passed to ASRPipeline constructor as a property
    ov_config["word_timestamps"] = True

    pipe = openvino_genai.ASRPipeline(args.model_dir, args.device, **ov_config)

    config = pipe.get_generation_config()

    # If language is known in advance it can be passed to the pipeline
    # In the form of "<|en|>" for Whisper models. Supported by multilingual models only
    # In the form of "English" for Qwen3-ASR models.
    config.language = "<|en|>"

    # Whisper models parameters. Ignored for Qwen3-ASR models
    config.task = "transcribe"
    config.return_timestamps = True
    config.word_timestamps = True

    # Pipeline expects normalized audio with Sample Rate of 16kHz
    raw_speech = read_wav(args.wav_file_path)
    result = pipe.generate(raw_speech, config)

    print(result)
    print_perf_metrics(result)

    if result.chunks:
        for chunk in result.chunks[0]:
            print(f"timestamps: [{chunk.start_ts:.2f}, {chunk.end_ts:.2f}] text: {chunk.text}")

    if result.words:
        for word in result.words[0]:
            print(f"[{word.start_ts:.2f}, {word.end_ts:.2f}]: {word.text}")


if "__main__" == __name__:
    main()
