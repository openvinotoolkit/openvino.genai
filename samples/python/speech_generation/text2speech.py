#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino as ov
import openvino_genai
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("text", help="Input text for which to generate speech")
    parser.add_argument("--speech_model_type", default="", choices=["", "speecht5_tts", "kokoro"],
                        help="Speech backend override (default: auto-detect from model)")
    parser.add_argument("--speaker_embedding_file_path", default=None,
                        help="Path to the binary file with a speaker embedding")
    parser.add_argument("--voice", default="", help="Kokoro voice id, e.g. af_heart")
    parser.add_argument("--language", default="", help="Kokoro language, e.g. en-us or en-gb")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier")
    parser.add_argument("--sample_rate", type=int, default=0,
                        help="Output wav sample rate override (default: 16000 for SpeechT5, 24000 for Kokoro)")
    parser.add_argument("--device", nargs="?", default="CPU", help="Device to run the model on (default: CPU)")
    args = parser.parse_args()

    # read speaker embedding from binary file
    speaker_embedding = None
    if args.speaker_embedding_file_path:
        speaker_embedding = np.fromfile(args.speaker_embedding_file_path, dtype=np.float32).reshape(1, 512)
        speaker_embedding = ov.Tensor(speaker_embedding)

    pipe = openvino_genai.Text2SpeechPipeline(args.model_dir, args.device)
    generation_properties = {}
    if args.speech_model_type:
        generation_properties["speech_model_type"] = args.speech_model_type
    if args.voice:
        generation_properties["voice"] = args.voice
    if args.language:
        generation_properties["language"] = args.language
    if args.speed != 1.0:
        generation_properties["speed"] = args.speed

    if speaker_embedding is not None:
        result = pipe.generate(args.text, speaker_embedding, **generation_properties)
    else:
        result = pipe.generate(args.text, None, **generation_properties)

    assert len(result.speeches) == 1, "Expected only one waveform for the requested input text"
    speech = result.speeches[0]
    speech_data = np.array(speech.data).reshape(-1)
    output_file_name = "output_audio.wav"
    if args.sample_rate > 0:
        sample_rate = args.sample_rate
    elif args.speech_model_type == "kokoro":
        sample_rate = 24000
    else:
        sample_rate = 16000
    sf.write(output_file_name, speech_data, samplerate=sample_rate)

    print("[Info] Text successfully converted to audio file \"", output_file_name, "\".")

    perf_metrics = result.perf_metrics;
    if perf_metrics.m_evaluated:
        print("\n\n=== Performance Summary ===")
        print("Throughput              : ", perf_metrics.throughput.mean, " samples/sec.")
        print("Total Generation Time   : ", perf_metrics.generate_duration.mean / 1000.0, " sec.")


if "__main__" == __name__:
    main()
