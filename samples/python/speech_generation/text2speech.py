#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino as ov
import openvino_genai
import soundfile as sf

def _load_speaker_embedding(file_path: str, shape):
    """Load a float32 binary and reshape it according to the given ov.Shape."""
    data = np.fromfile(file_path, dtype=np.float32)
    if data.size == 0:
        raise RuntimeError(f'Speaker embedding file is empty: {file_path}')

    return ov.Tensor(data.reshape(shape))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("text", help="Input text for speech generation")
    parser.add_argument("--speaker_embedding_file_path", default=None,
                        help="Path to the binary file with a speaker embedding. Required for Kokoro.")
    parser.add_argument("--language", default="", help="Optional language, e.g. en-us, en-gb, es, fr-fr, hi, it, pt-br")
    parser.add_argument("--device", nargs="?", default="CPU", help="Device to run the model on (default: CPU)")
    args = parser.parse_args()

    pipe = openvino_genai.Text2SpeechPipeline(args.model_dir, args.device)

    # Read speaker embedding using the model's expected shape.
    speaker_embedding = None
    if args.speaker_embedding_file_path:
        speaker_embedding = _load_speaker_embedding(args.speaker_embedding_file_path,
                                                    pipe.get_speaker_embedding_shape())

    generation_properties = {}
    language = args.language.strip().lower()

    if language:
        generation_properties["language"] = language

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
