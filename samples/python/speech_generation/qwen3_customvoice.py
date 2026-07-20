#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Qwen3-TTS CustomVoice sample.
#
# CustomVoice models synthesize speech using one of the model's built-in speaker
# identities. You select a voice with --speaker and, optionally, steer the
# delivery (tone, emotion, pace) with a natural-language --instruct description.
# No reference audio or speaker embedding is required.

import argparse

import numpy as np
import openvino_genai
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS CustomVoice generation")
    parser.add_argument("model_dir", help="Path to the Qwen3-TTS CustomVoice OpenVINO model directory")
    parser.add_argument("text", help="Input text to synthesize")
    parser.add_argument("--speaker", required=True, help="Built-in speaker name (for example: ryan)")
    parser.add_argument("--language", default="", help="Optional language (for example: english). Omit for auto.")
    parser.add_argument("--instruct", default="", help="Optional natural-language style instruction")
    parser.add_argument("--device", default="CPU", help="Device to run the model on (default: CPU)")
    args = parser.parse_args()

    pipe = openvino_genai.Text2SpeechPipeline(args.model_dir, args.device)

    generation_properties = {"speaker": args.speaker}
    if args.language:
        generation_properties["language"] = args.language
    if args.instruct:
        generation_properties["instruct"] = args.instruct

    # CustomVoice does not use an external speaker embedding.
    result = pipe.generate(args.text, **generation_properties)

    assert len(result.speeches) == 1, "Expected only one waveform for the requested input text"
    speech_data = np.array(result.speeches[0].data).reshape(-1)
    output_file_name = "output_audio.wav"
    sf.write(output_file_name, speech_data, samplerate=result.output_sample_rate)

    print(f'[Info] Text successfully converted to audio file "{output_file_name}".')

    perf_metrics = result.perf_metrics
    if perf_metrics.m_evaluated:
        print("\n\n=== Performance Summary ===")
        print("Throughput              : ", perf_metrics.throughput.mean, " samples/sec.")
        print("Total Generation Time   : ", perf_metrics.generate_duration.mean / 1000.0, " sec.")


if "__main__" == __name__:
    main()
