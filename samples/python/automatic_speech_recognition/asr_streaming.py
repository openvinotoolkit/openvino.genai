#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Streaming ASR with ASRStreamingSession (Qwen3-ASR, Voxtral Realtime).

Audio is pushed in --block-ms pieces from the microphone, or from a WAV file at real-time speed as a live microphone
would deliver it. After every push the available results are read and printed as a timeline. At the end the streaming
latency is compared with offline ASRPipeline.generate() over the same audio.

With --streams N, N sessions transcribe the same WAV file concurrently from N threads. On the PagedAttention backend
(--attention-backend PA) the pipeline batches their inference.
"""

import argparse
import threading
import time

import librosa
import numpy as np
import openvino_genai

SAMPLE_RATE = 16000
print_lock = threading.Lock()


class StreamReport:
    """Prints one timeline line per result and collects latencies of one stream."""

    def __init__(self, stream_id: int, chunk_sec: float, audio_start: float):
        self.stream_id = stream_id
        self.chunk_sec = chunk_sec
        self.audio_start = audio_start
        self.audio_sec = 0.0  # Audio pushed so far.
        self.latencies_ms = []
        self.compute_ms = 0.0  # Total time spent in read(), i.e. the inference cost of the whole stream.
        self.text = ""

    def add(self, result: openvino_genai.ASRStreamingResult, read_ms: float) -> None:
        # read_ms: time spent in ASRStreamingSession.read().
        # Latency: time from the moment the last sample of the chunk was spoken until read() returned its result.
        begin_sec = min(self.chunk_sec * len(self.latencies_ms), self.audio_sec)
        end_sec = self.audio_sec if result.is_final else min(begin_sec + self.chunk_sec, self.audio_sec)
        latency_ms = (time.perf_counter() - self.audio_start - end_sec) * 1000
        self.latencies_ms.append(latency_ms)
        self.compute_ms += read_ms
        if result.is_final:
            self.text = result.committed_text
        kind = "final" if result.is_final else "partial"
        with print_lock:
            print(
                f"[stream {self.stream_id}] {kind:<7} audio {begin_sec:5.1f}-{end_sec:5.1f} s | "
                f"read {read_ms:6.0f} ms | latency {latency_ms:6.0f} ms | "
                f'+"{result.new_committed_text}" [{result.partial_text}]'
            )


def transcribe(pipe, config, blocks, stream_id: int, realtime: bool) -> StreamReport:
    audio_start = time.perf_counter()
    report = StreamReport(stream_id, config.chunk_size_sec, audio_start)

    def read_available(session):
        """Reads all results that are due and adds them to the report."""
        while True:
            read_start = time.perf_counter()
            result = session.read()
            if result is None:
                break
            report.add(result, (time.perf_counter() - read_start) * 1000)

    # 1. Open a streaming session on the pipeline.
    session = openvino_genai.ASRStreamingSession(pipe, config)

    for block in blocks:
        report.audio_sec += len(block) / SAMPLE_RATE
        if realtime:
            # Simulate a live source: the block is delivered once it has been spoken.
            time.sleep(max(0.0, audio_start + report.audio_sec - time.perf_counter()))

        # 2. Feed audio. push() only buffers it.
        session.push(block)
        # 3. Read results. read() runs inference once a full chunk of audio is buffered, else returns None.
        read_available(session)

    # 4. Close input. The next read() calls transcribe the remaining audio; the last result has is_final == True.
    session.end()
    read_available(session)
    return report


def word_differences(text: str, reference: str) -> int:
    """Word-level edit distance: substituted, inserted and deleted words."""
    words, ref = text.split(), reference.split()
    row = list(range(len(ref) + 1))
    for i in range(1, len(words) + 1):
        diagonal, row[0] = row[0], i
        for j in range(1, len(ref) + 1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (words[i - 1] != ref[j - 1]))
    return row[len(ref)]


def print_summary(reports, audio_sec: float, offline_ms: float, offline_text: str) -> None:
    latencies = sorted(latency for report in reports for latency in report.latencies_ms)
    final_latency = sum(report.latencies_ms[-1] for report in reports) / len(reports)
    compute_ms = sum(report.compute_ms for report in reports) / len(reports)

    def percentile(p: float) -> float:
        return latencies[min(len(latencies) - 1, int(p * len(latencies)))]

    # Latency: offline generate() starts only after the speech has ended, so all of its time is latency.
    # Streaming runs most inference while the speech goes on; only the last chunks remain after it ends.
    print("\n=== Latency: time from end of speech to text ===")
    print(
        f"streaming, every result: p50 {percentile(0.5):6.0f} ms | p95 {percentile(0.95):6.0f} ms | "
        f"max {latencies[-1]:6.0f} ms"
    )
    print(f"streaming, final result: {final_latency:6.0f} ms")
    print(f"offline generate()     : {offline_ms:6.0f} ms")

    # Compute: total inference time of the whole audio, per stream. Above 1.0 s per 1 s of audio, streaming can't
    # keep up with a live source and its latency grows with every chunk.
    print("\n=== Compute: total inference time per stream ===")
    print(f"streaming, sum of read(): {compute_ms:6.0f} ms | {compute_ms / 1000 / audio_sec:4.2f} s per 1 s of audio")
    print(f"offline generate()      : {offline_ms:6.0f} ms | {offline_ms / 1000 / audio_sec:4.2f} s per 1 s of audio")

    print("\n=== Text ===")
    for report in reports:
        differences = word_differences(report.text, offline_text)
        print(
            f"[stream {report.stream_id}] {differences} of {len(offline_text.split())} words differ from offline: "
            f"{report.text}"
        )
    print(f"[offline] {offline_text}")


def blocks_of(audio: np.ndarray, block_size: int):
    for start in range(0, len(audio), block_size):
        yield audio[start : start + block_size]


def microphone_blocks(duration_sec: float, block_size: int, captured: list):
    import pyaudio

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=block_size)
    print("Microphone is live. Start speaking now.")
    try:
        for _ in range(int(duration_sec * SAMPLE_RATE) // block_size):
            data = stream.read(block_size, exception_on_overflow=False)
            block = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            captured.append(block)
            yield block
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--attention-backend", choices=["PA", "SDPA"], help="Attention backend of the pipeline")
    parser.add_argument("--chunk-size-sec", type=float, default=1.0, help="Audio duration between inferences")
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--window-chunk-num", type=int, default=8, help="Audio window in chunks, 0 = unbounded")
    parser.add_argument("--wav", help="WAV file. If omitted, microphone input is used")
    parser.add_argument("--duration", type=float, default=10.0, help="Microphone capture duration in seconds")
    parser.add_argument("--block-ms", type=int, default=500, help="Size of pushed audio blocks in milliseconds")
    parser.add_argument("--streams", type=int, default=1, help="Concurrent sessions over the same WAV file")
    args = parser.parse_args()

    properties = {"ATTENTION_BACKEND": args.attention_backend} if args.attention_backend else {}
    pipe = openvino_genai.ASRPipeline(args.model_dir, args.device, **properties)

    config = openvino_genai.ASRStreamingConfig()
    config.chunk_size_sec = args.chunk_size_sec
    config.unfixed_token_num = args.unfixed_token_num
    config.unfixed_chunk_num = args.unfixed_chunk_num
    config.window_chunk_num = args.window_chunk_num

    block_size = max(1, args.block_ms * SAMPLE_RATE // 1000)
    if args.wav is None:
        captured = []
        reports = [transcribe(pipe, config, microphone_blocks(args.duration, block_size, captured), 0, False)]
        audio = np.concatenate(captured)
    else:
        audio = librosa.load(args.wav, sr=SAMPLE_RATE, mono=True)[0].astype(np.float32)
        reports = [None] * args.streams

        def run(stream_id: int) -> None:
            reports[stream_id] = transcribe(pipe, config, blocks_of(audio, block_size), stream_id, True)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(args.streams)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if any(report is None for report in reports):
            raise RuntimeError("A streaming session failed, see the error above")

    # Offline baseline: the whole audio in one generate() call, which can only start once the speech has ended.
    offline_start = time.perf_counter()
    offline = pipe.generate(audio.tolist())
    offline_ms = (time.perf_counter() - offline_start) * 1000

    print_summary(reports, len(audio) / SAMPLE_RATE, offline_ms, offline.texts[0])


if __name__ == "__main__":
    main()
