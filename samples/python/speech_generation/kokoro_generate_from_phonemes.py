#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino_genai
import soundfile as sf

try:
    from kokoro import KPipeline
except ImportError as import_error:
    raise RuntimeError(
        "This sample requires Python Kokoro (which uses Misaki for G2P). "
        "Install it with `pip install kokoro` or add Kokoro to PYTHONPATH."
    ) from import_error


SAMPLE_TEXTS = {
    "en-us": "The sky above the port was the color of television, tuned to a dead channel.",
    "en-gb": "It's not like I'm using, Case heard someone say, as he shouldered his way through the crowd.",
    "es": "Los partidos políticos tradicionales compiten con los populismos y los movimientos asamblearios.",
    "fr-fr": "Le dromadaire resplendissant déambulait tranquillement dans les méandres en mastiquant de petites feuilles vernissées.",
    "hi": "ट्रांसपोर्टरों की हड़ताल लगातार पांचवें दिन जारी, दिसंबर से इलेक्ट्रॉनिक टोल कलेक्शनल सिस्टम.",
    "it": "Allora cominciava l'insonnia, o un dormiveglia peggiore dell'insonnia, che talvolta assumeva i caratteri dell'incubo.",
    "pt-br": "Elabora relatórios de acompanhamento cronológico para as diferentes unidades do Departamento que propõem contratos.",
    "ja": "「もしおれがただ偶然、そしてこうしようというつもりでなくここに立っているのなら、ちょっとばかり絶望するところだな」と、そんなことが彼の頭に思い浮かんだ。",
    "zh": "中國人民不信邪也不怕邪，不惹事也不怕事，任何外國不要指望我們會拿自己的核心利益做交易。",
}

DEFAULT_VOICES = {
    "en-us": "af_heart",
    "en-gb": "bf_emma",
    "es": "ef_dora",
    "fr-fr": "ff_siwis",
    "hi": "hf_alpha",
    "it": "if_sara",
    "pt-br": "pf_dora",
    "ja": "jf_alpha",
    "zh": "zf_xiaoxiao",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to Kokoro model directory")
    parser.add_argument(
        "--voice",
        default="",
        help="Optional Kokoro voice id. If not set, a language-specific default voice is used.",
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
    voice = args.voice or DEFAULT_VOICES.get(language, "af_heart")
    g2p_pipeline = KPipeline(lang_code=language, model=False, repo_id="hexgrad/Kokoro-82M")
    phonemes, misaki_tokens = g2p_pipeline.g2p(text)

    generation_properties = {
        "voice": voice,
        "language": language,
        "speed": args.speed,
    }

    print(f"[Info] Language={language}, voice={voice}")

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
        result = pipe.generate_from_tokens(ov_tokens, None, **generation_properties)
    else:
        if isinstance(phonemes, list):
            phoneme_chunks = phonemes
        elif isinstance(phonemes, str):
            phoneme_chunks = [phonemes]
        else:
            raise RuntimeError(f"Unsupported Misaki phoneme output type: {type(phonemes)!r}")
        print(f"[Info] Language={language}, using generate_from_phonemes with {len(phoneme_chunks)} chunk(s)")
        print("Generating from phonemes...")
        result = pipe.generate_from_phonemes(phoneme_chunks, None, **generation_properties)

    save_waveform(result, args.output)


if "__main__" == __name__:
    main()
