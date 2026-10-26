#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import time
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
    parser.add_argument("--profile", action="store_true", help="Show OpenVINO operation profiling inside infer()")
    args = parser.parse_args()

    audio, _ = librosa.load(args.audio_file, sr=SAMPLE_RATE, mono=True)
    properties = {"PERF_COUNT": True} if args.profile else {}
    pipe = ov_genai.ASRPipeline(args.model_dir, args.device, **properties)
    stream = pipe.create_stream()

    def push_audio():
        for start in range(0, len(audio), FRAME_SAMPLES):
            stream.push_audio(audio[start : start + FRAME_SAMPLES].tolist())
        stream.end_audio()

    generation_start = time.perf_counter()
    last_report_time = generation_start
    last_reported_tokens = 0
    last_stage_durations = stream.get_stage_durations()

    def format_stage_durations(durations, num_tokens):
        if num_tokens == 0:
            return "Stage timings: no new generation steps"
        return (
            "ms/step: "
            f"audio infer={durations['audio_inference'] / num_tokens:.2f} | "
            f"text infer={durations['text_inference'] / num_tokens:.2f} | "
            f"audio KV={durations['audio_kv_cache'] / num_tokens:.2f} | "
            f"text KV={durations['text_kv_cache'] / num_tokens:.2f}"
        )

    def format_cache_windows():
        windows = stream.get_kv_cache_window_sizes()
        return " | ".join(
            f"{component} window: {used}/{maximum} ({100 * used / maximum:.1f}%)"
            for component, (used, maximum) in windows.items()
        )

    def report_operation_durations(durations, num_tokens):
        if not args.profile or num_tokens == 0:
            return
        for component in ("audio", "text"):
            groups = sorted(
                ((stage, duration) for stage, duration in durations.items() if stage.startswith(f"{component}_op/")),
                key=lambda item: item[1],
                reverse=True,
            )
            if not groups:
                raise RuntimeError(f"No executed operation profiling counters returned for {component} inference")
            print(
                f"  {component} ops plugin ms (CPU running averages, top 8): "
                + " | ".join(f"{stage.split('/', 1)[1]}={duration:.2f}" for stage, duration in groups[:8]),
                file=sys.stderr,
                flush=True,
            )
            nodes = sorted(
                ((stage, duration) for stage, duration in durations.items() if stage.startswith(f"{component}_node/")),
                key=lambda item: item[1],
                reverse=True,
            )
            for stage, duration in nodes[:3]:
                print(
                    f"    Broadcast {duration:.3f} plugin ms: {stage.split('/', 1)[1]}",
                    file=sys.stderr,
                    flush=True,
                )

    def report_delta(delta):
        nonlocal last_report_time, last_reported_tokens, last_stage_durations
        now = time.perf_counter()
        token_count = stream.get_num_generated_tokens()
        elapsed = now - generation_start
        interval = now - last_report_time
        interval_tokens = token_count - last_reported_tokens
        average_rate = token_count / elapsed if elapsed > 0 else 0.0
        interval_rate = interval_tokens / interval if interval > 0 else 0.0
        stage_durations = stream.get_stage_durations()
        interval_durations = {
            stage: stage_durations[stage] - last_stage_durations[stage]
            for stage in ("audio_inference", "text_inference", "audio_kv_cache", "text_kv_cache")
        }

        print(delta, end="", flush=True)
        print(
            f"\n[{elapsed:6.1f}s] {token_count} tokens | "
            f"{interval_rate:.1f} tok/s interval | {average_rate:.1f} tok/s avg | "
            f"KV cache: {stream.get_kv_cache_size() / (1024**2):.1f} MiB | {format_cache_windows()}\n"
            f"  interval {format_stage_durations(interval_durations, interval_tokens)}",
            file=sys.stderr,
            flush=True,
        )
        report_operation_durations(stage_durations, interval_tokens)
        last_report_time = now
        last_reported_tokens = token_count
        last_stage_durations = stage_durations
        return ov_genai.StreamingStatus.RUNNING

    with ThreadPoolExecutor(max_workers=1) as executor:
        producer = executor.submit(push_audio)
        result = stream.generate(report_delta)
        producer.result()

    elapsed = time.perf_counter() - generation_start
    token_count = stream.get_num_generated_tokens()
    average_rate = token_count / elapsed if elapsed > 0 else 0.0
    print(
        f"\n[{elapsed:6.1f}s] Complete: {token_count} tokens | {average_rate:.1f} tok/s avg | "
        f"KV cache: {stream.get_kv_cache_size() / (1024**2):.1f} MiB | {format_cache_windows()}\n"
        f"  overall {format_stage_durations(stream.get_stage_durations(), token_count)}",
        file=sys.stderr,
        flush=True,
    )
    report_operation_durations(stream.get_stage_durations(), token_count)
    print("\nFinal:", result.texts[0])


if __name__ == "__main__":
    main()
