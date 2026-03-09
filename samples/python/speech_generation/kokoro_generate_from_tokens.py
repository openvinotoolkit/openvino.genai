#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino_genai
import soundfile as sf

try:
    from misaki import en
except ImportError as import_error:
    raise RuntimeError(
        "This sample requires Python Misaki. Install it with `pip install misaki` "
        "or add Misaki to PYTHONPATH."
    ) from import_error


def _speech_token_type():
    token_type = getattr(openvino_genai, "SpeechToken", None)
    if token_type is None:
        raise RuntimeError("openvino_genai.SpeechToken is not available in this build")
    return token_type


def misaki_tokens_to_ov_tokens(misaki_tokens):
    token_type = _speech_token_type()
    ov_tokens = []
    for token in misaki_tokens:
        if token.phonemes is None and not token.whitespace:
            continue
        ov_tokens.append(
            token_type(
                phonemes="" if token.phonemes is None else token.phonemes,
                whitespace=bool(token.whitespace),
                text=token.text,
            )
        )
    return ov_tokens


def save_waveform(result, output_file: str):
    assert len(result.speeches) == 1, "Expected exactly one speech output"
    sf.write(output_file, np.array(result.speeches[0].data).reshape(-1), samplerate=result.output_sample_rate)
    print(f'[Info] Audio saved to "{output_file}"')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to Kokoro model directory")
    parser.add_argument("text", help="Input text to phonemize with Misaki and synthesize with OpenVINO GenAI")
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice id, e.g. af_heart")
    parser.add_argument("--language", default="en-us", help="Kokoro language, e.g. en-us or en-gb")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier")
    parser.add_argument("--output", default="output_audio.wav", help="Output WAV file path")
    parser.add_argument("--device", nargs="?", default="CPU", help="Device to run the model on (default: CPU)")
    args = parser.parse_args()

    pipe = openvino_genai.Text2SpeechPipeline(args.model_dir, args.device)

    language = args.language.lower()
    if language not in ("en-us", "en-gb"):
        raise ValueError("This sample currently supports Misaki English G2P only: use --language en-us or en-gb")
    g2p = en.G2P(trf=False, british=(language == "en-gb"), fallback=None, unk="")
    _, misaki_tokens = g2p(args.text)
    ov_tokens = misaki_tokens_to_ov_tokens(misaki_tokens)
    if not ov_tokens:
        raise RuntimeError("Misaki produced no tokens with phonemes for the provided input text")

    print(f"[Info] Converted {len(misaki_tokens)} Misaki tokens to {len(ov_tokens)} OpenVINO SpeechToken entries")

    generation_properties = {
        "voice": args.voice,
        "language": language,
        "speed": args.speed,
    }

    result = pipe.generate_from_tokens(ov_tokens, None, **generation_properties)
    save_waveform(result, args.output)


if "__main__" == __name__:
    main()
