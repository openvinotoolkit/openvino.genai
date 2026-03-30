#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino as ov
import openvino_genai
import soundfile as sf

try:
    from kokoro import KPipeline
except ImportError as import_error:
    raise RuntimeError(
        "This sample requires Python Kokoro (which uses Misaki for G2P). "
        "Install it with `pip install kokoro`"
    ) from import_error


SAMPLE_TEXTS = {
    "en-us": "On a bright morning, a gentle breeze carried the scent of fresh coffee and new possibilities.",
    "en-gb": "As the sun peeked through the clouds, everything felt quietly hopeful and full of promise.",
    "es": "En una mañana luminosa, el aire fresco traía consigo nuevas oportunidades y una sensación de calma.",
    "fr-fr": "Par une matinée ensoleillée, une douce brise apportait une sensation de joie et de renouveau.",
    "hi": "सुबह की हल्की धूप और ठंडी हवा मन में नई उम्मीद और शांति का एहसास भर देती है।",
    "it": "In una mattina luminosa, una brezza leggera portava con sé un senso di serenità e nuove possibilità.",
    "pt-br": "Em uma manhã ensolarada, uma brisa suave trazia uma sensação de tranquilidade e novas oportunidades.",
    "ja": "明るい朝、やさしい風が吹き、新しい一日への期待が静かに広がっていった。",
    "zh": "在一个阳光明媚的早晨，轻柔的微风带来了宁静与新的希望。",
}

def _speech_token_type():
    token_type = getattr(openvino_genai, "SpeechToken", None)
    if token_type is None:
        raise RuntimeError("openvino_genai.SpeechToken is not available in this build")
    return token_type


def misaki_tokens_to_ov_tokens(misaki_tokens):
    token_type = _speech_token_type()
    ov_tokens = []
    for token in misaki_tokens:
        token_phonemes = getattr(token, "phonemes", None)
        token_whitespace = getattr(token, "whitespace", "")
        if token_phonemes is None and not token_whitespace:
            continue
        ov_tokens.append(
            token_type(
                phonemes="" if token_phonemes is None else token_phonemes,
                whitespace=bool(token_whitespace),
                text=getattr(token, "text", ""),
            )
        )
    return ov_tokens


def save_waveform(result, output_file: str):
    assert len(result.speeches) == 1, "Expected exactly one speech output"
    sf.write(output_file, np.array(result.speeches[0].data).reshape(-1), samplerate=result.output_sample_rate)
    print(f'[Info] Audio saved to "{output_file}"')


def load_kokoro_embedding(file_path: str, shape):
    """Load a float32 binary and reshape it according to the given ov.Shape."""
    data = np.fromfile(file_path, dtype=np.float32)
    if data.size == 0:
        raise RuntimeError(f'Speaker embedding file is empty: {file_path}')
    return ov.Tensor(data.reshape(shape))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to Kokoro model directory")
    parser.add_argument(
        "--speaker_embedding_file_path",
        required=True,
        help="Path to a prepared Kokoro speaker embedding binary",
    )
    parser.add_argument(
        "--language",
        default="en-us",
        choices=sorted(SAMPLE_TEXTS.keys()),
        help="Language used to choose both Misaki G2P and a predefined sample text",
    )
    parser.add_argument(
        "--api",
        default="auto",
        choices=["auto", "tokens", "phonemes"],
        help="OV GenAI API path: auto prefers tokens if available, otherwise phonemes",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier")
    parser.add_argument("--output", default="output_audio.wav", help="Output WAV file path")
    parser.add_argument("--device", nargs="?", default="CPU", help="Device to run the model on (default: CPU)")
    args = parser.parse_args()

    pipe = openvino_genai.Text2SpeechPipeline(args.model_dir, args.device)

    language = args.language.lower()
    text = SAMPLE_TEXTS[language]
    embedding_dim = pipe.get_speaker_embedding_shape()
    speaker_embedding = load_kokoro_embedding(args.speaker_embedding_file_path, embedding_dim)
    g2p_pipeline = KPipeline(lang_code=language, model=False, repo_id="hexgrad/Kokoro-82M")
    phonemes, misaki_tokens = g2p_pipeline.g2p(text)

    generation_properties = {
        "language": language,
        "speed": args.speed,
    }

    print(f"[Info] Language={language}, speaker_embedding={args.speaker_embedding_file_path}")

    use_tokens = args.api == "tokens"
    if args.api == "auto":
        use_tokens = misaki_tokens is not None

    if use_tokens:
        if misaki_tokens is None:
            raise RuntimeError(
                f"Misaki G2P for language '{language}' did not return token objects. Use --api phonemes or --api auto."
            )
        ov_tokens = misaki_tokens_to_ov_tokens(misaki_tokens)
        if not ov_tokens:
            raise RuntimeError("Misaki produced no usable tokens for generate_from_tokens")
        print(f"[Info] Language={language}, using generate_from_tokens with {len(ov_tokens)} SpeechToken entries")
        print("Generating from tokens...")
        result = pipe.generate_from_tokens(ov_tokens, speaker_embedding, **generation_properties)
    else:
        if isinstance(phonemes, list):
            phoneme_chunks = phonemes
        elif isinstance(phonemes, str):
            phoneme_chunks = [phonemes]
        else:
            raise RuntimeError(f"Unsupported Misaki phoneme output type: {type(phonemes)!r}")
        print(f"[Info] Language={language}, using generate_from_phonemes with {len(phoneme_chunks)} chunk(s)")
        print("Generating from phonemes...")
        result = pipe.generate_from_phonemes(phoneme_chunks, speaker_embedding, **generation_properties)

    save_waveform(result, args.output)


if "__main__" == __name__:
    main()
