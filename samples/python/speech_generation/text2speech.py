#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino as ov
import openvino_genai
import soundfile as sf

DEFAULT_KOKORO_VOICES = {
    "en-us": "af_heart",
    "en-gb": "bf_emma",
    "es": "ef_dora",
    "fr-fr": "ff_siwis",
    "hi": "hf_alpha",
    "it": "if_sara",
    "pt-br": "pf_dora",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("text", help="Input text for speech generation")
    parser.add_argument("--speaker_embedding_file_path", default=None,
                        help="Path to the binary file with a speaker embedding")
    parser.add_argument("--voice", default="", help="Optional voice id (required by Kokoro when no speaker embedding)")
    parser.add_argument("--language", default="", help="Optional language, e.g. en-us, en-gb, es, fr-fr, hi, it, pt-br")
    parser.add_argument("--device", nargs="?", default="CPU", help="Device to run the model on (default: CPU)")
    args = parser.parse_args()

    # read speaker embedding from binary file
    speaker_embedding = None
    if args.speaker_embedding_file_path:
        speaker_embedding = np.fromfile(args.speaker_embedding_file_path, dtype=np.float32).reshape(1, 512)
        speaker_embedding = ov.Tensor(speaker_embedding)

    pipe = openvino_genai.Text2SpeechPipeline(args.model_dir, args.device)

    generation_properties = {}
    language = args.language.strip().lower()
    voice = args.voice.strip()

    if language:
        generation_properties["language"] = language

    if voice:
        generation_properties["voice"] = voice
    elif speaker_embedding is None and language in DEFAULT_KOKORO_VOICES:
        generation_properties["voice"] = DEFAULT_KOKORO_VOICES[language]
        print(f"[Info] No --voice provided. Using default Kokoro voice '{generation_properties['voice']}' for language '{language}'.")
    
    result = pipe.generate(args.text, speaker_embedding, **generation_properties)

    assert len(result.speeches) == 1, "Expected only one waveform for the requested input text"
    speech = result.speeches[0]
    speech_data = np.array(speech.data).reshape(-1)
    output_file_name = "output_audio.wav"
    sf.write(output_file_name, speech_data, samplerate=result.output_sample_rate)

    print("[Info] Text successfully converted to audio file \"", output_file_name, "\".")

    perf_metrics = result.perf_metrics;
    if perf_metrics.m_evaluated:
        print("\n\n=== Performance Summary ===")
        print("Throughput              : ", perf_metrics.throughput.mean, " samples/sec.")
        print("Total Generation Time   : ", perf_metrics.generate_duration.mean / 1000.0, " sec.")


if "__main__" == __name__:
    main()
