#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino as ov
import openvino_genai
import soundfile as sf


def _load_kokoro_embedding(file_path: str, shape):
    """Load a float32 binary and reshape it according to the given ov.Shape."""
    data = np.fromfile(file_path, dtype=np.float32)
    if data.size == 0:
        raise RuntimeError(f"Speaker embedding file is empty: {file_path}")
    return ov.Tensor(data.reshape(shape))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to Kokoro model directory")
    parser.add_argument("text", help="Input text for speech generation")
    parser.add_argument(
        "--speaker_embedding_file_path", required=True, help="Path to a prepared Kokoro speaker embedding binary"
    )
    parser.add_argument(
        "--language",
        default="en-us",
        choices=["en-us", "en-gb"],
        help="Kokoro language variant for this sample (en-us or en-gb)",
    )
    parser.add_argument("--device", default="CPU", help="Device to run the model on (default: CPU)")
    parser.add_argument(
        "--phonemize_fallback_model_dir",
        default=None,
        help=(
            "Optional OpenVINO fallback G2P model directory. "
            "If set, Kokoro phonemize fallback uses this model. "
            "If omitted, fallback defaults to espeak-ng."
        ),
    )
    parser.add_argument("--output", default="output_audio.wav", help="Output WAV file path")
    args = parser.parse_args()

    pipe = openvino_genai.Text2SpeechPipeline(args.model_dir, args.device)
    speaker_embedding = _load_kokoro_embedding(args.speaker_embedding_file_path, pipe.get_speaker_embedding_shape())

    config = pipe.get_generation_config()
    config.language = args.language
    config.phonemize_fallback_model_dir = args.phonemize_fallback_model_dir
    pipe.set_generation_config(config)

    result = pipe.generate(args.text, speaker_embedding)
    assert len(result.speeches) == 1, "Expected one waveform"

    speech = result.speeches[0]
    speech_data = np.array(speech.data).reshape(-1)
    sf.write(args.output, speech_data, samplerate=result.output_sample_rate)

    fallback_mode = (
        f"OpenVINO fallback ({args.phonemize_fallback_model_dir})"
        if args.phonemize_fallback_model_dir
        else "espeak-ng fallback"
    )
    print(f"[Info] Generated '{args.output}' using {fallback_mode} during phonemize/G2P fallback.")


if __name__ == "__main__":
    main()
