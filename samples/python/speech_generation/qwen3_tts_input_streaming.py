#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-TTS input streaming with Text2SpeechStreamingSession (CustomVoice, VoiceDesign, Base).

The text is spoken while it "arrives", the way an LLM reply would: a producer thread pushes it word by word at a
chosen rate (--tokens_per_second) and the main thread reads the audio as soon as it is ready. After every push the
available results are read and printed as a timeline. With --compare the same text is also generated in one go with
Text2SpeechPipeline.generate() and the two are compared.

Notes:
  * Input streaming needs the SDPA backend, so the pipeline is created with ATTENTION_BACKEND="SDPA".
  * One session at a time per pipeline, and generate() cannot run while a session is alive.
  * A session produces audio of the same length and content as generate() for the whole text with
    non_streaming_mode=False (the layout a session always uses); the default non_streaming_mode=True of generate()
    puts the whole text into the prompt up front, which a session cannot do.
"""

import argparse
import re
import threading
import time

import numpy as np
import openvino as ov
import openvino_genai
import soundfile as sf

SAMPLE_RATE = 24000


def load_speaker_embedding(file_path, shape):
    data = np.fromfile(file_path, dtype="<f4")
    if data.size == 0:
        raise RuntimeError(f"Speaker embedding file is empty: {file_path}")
    return ov.Tensor(data.reshape(shape))


def load_ref_audio(file_path):
    audio, sample_rate = sf.read(file_path, dtype="float32", always_2d=False)
    if int(sample_rate) != SAMPLE_RATE:
        raise RuntimeError(
            f"Reference audio sample rate is {int(sample_rate)} Hz, but {SAMPLE_RATE} Hz is required. "
            "OV GenAI does not resample reference audio."
        )
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype(np.float32)
    return ov.Tensor(np.ascontiguousarray(audio, dtype=np.float32))


def split_into_tokens(text):
    """Pieces like an LLM streams them: a word with the whitespace in front of it."""
    return re.findall(r"\s*\S+|\s+$", text)


def push_text(session, tokens, tokens_per_second, push_times):
    """Producer: pushes the tokens at a fixed rate, then closes the text."""
    start = time.perf_counter()
    for index, token in enumerate(tokens):
        if tokens_per_second > 0:
            time.sleep(max(0.0, start + index / tokens_per_second - time.perf_counter()))
        push_times.append(time.perf_counter())
        session.push(token)
    session.end()


def run_streaming(pipe, text, speaker_embedding, generation_properties, tokens_per_second):
    tokens = split_into_tokens(text)
    # 1. Open a session on the pipeline. The voice is configured exactly as for generate().
    session = openvino_genai.Text2SpeechStreamingSession(pipe, speaker_embedding, **generation_properties)

    push_times = []
    producer = threading.Thread(target=push_text, args=(session, tokens, tokens_per_second, push_times))
    audio_start = time.perf_counter()
    producer.start()  # 2. Feed text from another thread. push() only buffers it.

    chunks = []
    first_audio_time = None
    sample_rate = SAMPLE_RATE
    finish_reason = None
    while True:
        # 3. Read audio. read() runs inference until a chunk is ready, or returns None when it has to wait for text.
        read_start = time.perf_counter()
        result = session.read()
        if result is None:
            time.sleep(0.002)  # nothing to return yet: wait for more text
            continue
        now = time.perf_counter()
        read_ms = (now - read_start) * 1000
        if first_audio_time is None and result.speech.get_size() > 0:
            first_audio_time = now
        samples = np.array(result.speech.data, dtype=np.float32).reshape(-1)
        chunks.append(samples)
        sample_rate = result.output_sample_rate
        kind = "final  " if result.is_final else "partial"
        print(
            f"[{now - audio_start:6.2f} s] {kind} audio {result.start_sample / sample_rate:6.2f}-"
            f"{(result.start_sample + samples.size) / sample_rate:6.2f} s | read {read_ms:6.0f} ms | "
            f"words pushed {len(push_times)}/{len(tokens)}"
        )
        if result.is_final:
            finish_reason = result.finish_reason
            break
    producer.join()

    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    ttfa = None if first_audio_time is None or not push_times else first_audio_time - push_times[0]
    return audio, sample_rate, ttfa, time.perf_counter() - audio_start, finish_reason


def print_summary(audio, sample_rate, ttfa, elapsed, finish_reason, tokens_per_second):
    audio_seconds = audio.size / sample_rate
    print("\n=== Summary ===")
    print(f"audio              : {audio_seconds:.2f} s ({audio.size} samples), finish reason {finish_reason}")
    if ttfa is not None:
        print(f"time to first audio: {ttfa * 1000:.0f} ms after the first word was pushed")
    print(f"wall time          : {elapsed:.2f} s" + (f" (text paced at {tokens_per_second} words/s)" if tokens_per_second > 0 else ""))


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_dir", help="Path to a Qwen3-TTS OpenVINO model directory (Base, CustomVoice or VoiceDesign)")
    parser.add_argument("text", help="Text to speak; it is pushed word by word")
    parser.add_argument("--speaker", default="", help="CustomVoice: built-in speaker name (for example: ryan)")
    parser.add_argument("--instruct", default="", help="VoiceDesign: voice description; CustomVoice: optional style instruction")
    parser.add_argument("--language", default="", help="Optional language (for example: english). Omit for auto.")
    parser.add_argument("--speaker_embedding_file_path", default=None, help="Base: speaker embedding (.bin, see qwen3_tts.py)")
    parser.add_argument("--ref_audio_wav_path", default=None, help="Base: reference audio WAV (24000 Hz)")
    parser.add_argument("--ref_text", default="", help="Base: reference transcript; enables ICL voice cloning")
    parser.add_argument("--device", default="CPU", help="Device to run the model on")
    parser.add_argument("--tokens_per_second", type=float, default=15.0, help="How fast the text is pushed, in words per second; 0 pushes it all at once")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Optional cap on the generated codec frames (12.5 per second of audio)")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed (42 with --compare when not given)")
    parser.add_argument("--compare", action="store_true", help="Also run generate() on the whole text and compare the audio")
    parser.add_argument("--output_wav_path", default="output_streaming_audio.wav", help="Output WAV path")
    return parser


def main():
    args = build_parser().parse_args()

    pipeline_kwargs = {"ATTENTION_BACKEND": "SDPA"}
    if "NPU" in args.device.upper() or "GPU" in args.device.upper():
        pipeline_kwargs["CACHE_DIR"] = "qwen3_tts_cache_dir"
    pipe = openvino_genai.Text2SpeechPipeline(args.model_dir, args.device, **pipeline_kwargs)

    generation_properties = {}
    for name, value in (("speaker", args.speaker), ("instruct", args.instruct), ("language", args.language)):
        if value:
            generation_properties[name] = value
    if args.ref_text.strip():
        generation_properties["ref_text"] = args.ref_text
    if args.ref_audio_wav_path:
        generation_properties["ref_audio"] = load_ref_audio(args.ref_audio_wav_path)
    if args.max_new_tokens is not None:
        generation_properties["max_new_tokens"] = args.max_new_tokens
    if args.seed is not None:
        generation_properties["rng_seed"] = args.seed
    elif args.compare:
        # Both runs have to draw the same random numbers to be comparable.
        generation_properties["rng_seed"] = 42
    speaker_embedding = None
    if args.speaker_embedding_file_path:
        speaker_embedding = load_speaker_embedding(args.speaker_embedding_file_path, pipe.get_speaker_embedding_shape())

    audio, sample_rate, ttfa, elapsed, finish_reason = run_streaming(
        pipe, args.text, speaker_embedding, generation_properties, args.tokens_per_second
    )
    sf.write(args.output_wav_path, audio, samplerate=sample_rate, subtype="FLOAT")
    print(f'[Info] Audio written to "{args.output_wav_path}".')
    print_summary(audio, sample_rate, ttfa, elapsed, finish_reason, args.tokens_per_second)

    if args.compare:
        # The session's layout: the text arrives over time, so generate() has to use the streaming layout as well.
        offline_start = time.perf_counter()
        offline = pipe.generate(args.text, speaker_embedding, non_streaming_mode=False, **generation_properties)
        offline_seconds = time.perf_counter() - offline_start
        reference = np.array(offline.speeches[0].data, dtype=np.float32).reshape(-1)
        print("\n=== Comparison with generate(non_streaming_mode=False) on the whole text ===")
        print(f"generate() took {offline_seconds:.2f} s for {reference.size / sample_rate:.2f} s of audio")
        if reference.size == audio.size and np.array_equal(reference, audio):
            print("the audio is identical")
        elif reference.size < audio.size:
            print(
                f"the audio differs: generate() ended the utterance earlier ({reference.size} samples) than the session "
                f"({audio.size}). generate() lets the model end the utterance at any step; a session only after the text "
                "has ended, so it cannot cut the text short."
            )
        else:
            print(f"the audio differs ({audio.size} vs {reference.size} samples), which is not expected.")


if __name__ == "__main__":
    main()
