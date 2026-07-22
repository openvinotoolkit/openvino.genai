# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import numpy as np

SPEECHT5_SPEAKER_EMB_SHAPE = (1, 512)
KOKORO_SPEAKER_EMB_SHAPE = (510, 1, 256)
DEFAULT_KOKORO_VOICE = "af_heart"
KOKORO_SAMPLE_RATE = 24000
OMNI_TTS_SAMPLE_RATE = 24000
DEFAULT_TTS_SAMPLE_RATE = 16000
DEFAULT_OMNI_SPEAKER = "Ethan"


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


def resolve_kokoro_speaker_embedding(model_path, speech_voice="", speaker_embeddings=None, strict=False):
    if speaker_embeddings is not None:
        return speaker_embeddings

    selected_voice = speech_voice.strip() if isinstance(speech_voice, str) else ""
    if not selected_voice:
        selected_voice = DEFAULT_KOKORO_VOICE

    if model_path is None:
        if strict:
            raise RuntimeError("Kokoro model path is required to resolve speaker embedding")
        return None

    voice_file = Path(model_path) / "voices" / f"{selected_voice}.bin"
    if voice_file.exists():
        from llm_bench_utils.model_utils import get_speaker_embeddings

        return get_speaker_embeddings(str(voice_file), expected_shape=KOKORO_SPEAKER_EMB_SHAPE)

    if strict:
        raise RuntimeError(
            f"Kokoro voice file not found: {voice_file}. "
            f"Pass --speaker_embeddings directly or ensure '{selected_voice}' exists under {Path(model_path) / 'voices'}/."
        )

    return None


def get_tts_sample_rate(args):
    if args.get("is_omni_model", False):
        return OMNI_TTS_SAMPLE_RATE
    if args.get("is_kokoro_model", False):
        return KOKORO_SAMPLE_RATE
    return DEFAULT_TTS_SAMPLE_RATE


def extract_audio_array(output):
    try:
        import torch

        if isinstance(output, torch.Tensor):
            return output.detach().cpu().reshape(-1).numpy()
    except ImportError:
        pass

    if hasattr(output, "data") and not isinstance(output, (list, tuple, np.ndarray)):
        try:
            return np.asarray(output.data, dtype=np.float32).reshape(-1)
        except Exception:
            pass

    if hasattr(output, "numpy"):
        return output.numpy().reshape(-1)

    return np.asarray(output, dtype=np.float32).reshape(-1)


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


def resolve_omni_generation_settings(args):
    speaker = (args.get("speech_voice") or "").strip() or DEFAULT_OMNI_SPEAKER
    max_new_tokens = int(args["infer_count"]) if args.get("infer_count") is not None else None
    num_beams = args.get("num_beams", 1) or 1
    seed = int(args.get("seed", 0) or 0)
    return {
        "speaker": speaker,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "seed": seed,
    }


def kokoro_generate_from_preprocessed(model, preprocessed_input, args):
    if hasattr(model, "generate_from_preprocessed"):
        generation_result = model.generate_from_preprocessed(preprocessed_input)
        return extract_audio_array(generation_result)

    raise RuntimeError("Kokoro model wrapper must provide generate_from_preprocessed")
