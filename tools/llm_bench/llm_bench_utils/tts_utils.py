# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import numpy as np
import openvino as ov

SPEECHT5_SPEAKER_EMB_SHAPE = (1, 512)
KOKORO_SPEAKER_EMB_SHAPE = (510, 1, 256)
DEFAULT_KOKORO_VOICE = "af_heart"
KOKORO_SAMPLE_RATE = 24000
DEFAULT_TTS_SAMPLE_RATE = 16000


def is_kokoro_model_id(model_id_or_path):
    model_str = str(model_id_or_path)
    # Fast path: model ID or directory name contains "kokoro".
    if "kokoro" in model_str.lower():
        return True
    # Fallback for local exports whose directory name doesn't include "kokoro":
    # require a voices/ subdirectory containing at least one canonical voice file.
    model_path = Path(model_str)
    voices_dir = model_path / "voices"
    if model_path.is_dir() and voices_dir.is_dir():
        if (
            (voices_dir / "af_heart.bin").exists()
            or (voices_dir / "af_heart.pt").exists()
            or (voices_dir / "tiny_voice.bin").exists()
            or (voices_dir / "tiny_voice.pt").exists()
        ):
            return True
    return False


def normalize_kokoro_lang_code(language):
    if not isinstance(language, str) or language.strip() == "":
        return "a"

    normalized = language.strip().lower()
    lang_map = {
        "en-us": "a",
        "en-gb": "b",
        "es": "e",
        "fr-fr": "f",
        "hi": "h",
        "it": "i",
        "pt-br": "p",
        "ja": "j",
        "zh": "z",
    }
    if normalized in lang_map:
        return lang_map[normalized]
    if len(normalized) == 1:
        return normalized

    raise ValueError(
        f"Unsupported Kokoro language '{language}'. Use one of: en-us, en-gb, es, fr-fr, hi, it, pt-br, ja, zh."
    )


def get_tts_sample_rate(args):
    return KOKORO_SAMPLE_RATE if args.get("is_kokoro_model", False) else DEFAULT_TTS_SAMPLE_RATE


def extract_audio_array(output):
    try:
        import torch

        if isinstance(output, torch.Tensor):
            return output.detach().cpu().reshape(-1).numpy()
    except ImportError:
        pass

    if hasattr(output, "numpy"):
        return output.numpy().reshape(-1)

    return np.asarray(output, dtype=np.float32).reshape(-1)


def kokoro_generate_once(model, input_text, args, use_genai):
    speaker_embeddings = args.get("speaker_embeddings")
    speech_language = args.get("speech_language", "")
    speech_voice = args.get("speech_voice", "")

    if use_genai:
        # Note: For GenAI Kokoro model, the speaker embedding is passed here, even if user specified --speech_voice.
        speaker_embedding_np = (
            speaker_embeddings.detach().cpu().numpy()
            if hasattr(speaker_embeddings, "detach")
            else np.asarray(speaker_embeddings)
        )
        generation_result = model.generate(
            input_text,
            speaker_embedding=ov.Tensor(speaker_embedding_np),
            language=speech_language,
        )
        if hasattr(generation_result, "speeches"):
            return extract_audio_array(generation_result.speeches[0].data)
        return extract_audio_array(generation_result)

    generation_result = model.generate(
        input_text,
        speaker_embeddings=speaker_embeddings,
        language=speech_language,
        voice=speech_voice,
    )
    return extract_audio_array(generation_result)


def kokoro_preprocess_once(model, input_text, args):
    speaker_embeddings = args.get("speaker_embeddings")
    speech_language = args.get("speech_language", "")
    speech_voice = args.get("speech_voice", "")

    if hasattr(model, "preprocess_input"):
        return model.preprocess_input(
            input_text,
            speaker_embeddings=speaker_embeddings,
            language=speech_language,
            voice=speech_voice,
        )
    return input_text


def kokoro_generate_from_preprocessed(model, preprocessed_input, args):
    if hasattr(model, "generate_from_preprocessed"):
        generation_result = model.generate_from_preprocessed(preprocessed_input)
        return extract_audio_array(generation_result)

    return kokoro_generate_once(model, preprocessed_input, args, use_genai=False)
