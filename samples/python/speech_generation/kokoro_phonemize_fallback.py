#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino_genai
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to Kokoro model directory")
    parser.add_argument("text", help="Input text for speech generation")
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice id")
    parser.add_argument("--language", default="en-us", help="Kokoro language variant (en-us or en-gb)")
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

    config = pipe.get_generation_config()
    config.voice = args.voice
    config.language = args.language
    config.phonemize_fallback_model_dir = args.phonemize_fallback_model_dir
    pipe.set_generation_config(config)

    result = pipe.generate(args.text, None)
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
