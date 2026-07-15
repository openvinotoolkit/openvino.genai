# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Union

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from importlib.resources import files
from tqdm import tqdm

from .registry import register_evaluator, BaseEvaluator
from .tts_similarity import (
    ScoringConfig,
    TTSSimilarityEvaluator,
    linear_distance_score,
    normalize_text,
    safe_float,
)

PROMPTS_FILE = "speech_generation_prompts.yaml"
OMNI_PROMPTS_FILE = "text_prompts.yaml"
DEFAULT_SPEAKER_EMBEDDING_REPO_ID = "Xenova/cmu-arctic-xvectors-extracted"
DEFAULT_SPEAKER_EMBEDDING_FILENAME = "cmu_us_slt_arctic-wav-arctic_a0508.bin"
SPEECHT5_SPEAKER_EMB_SHAPE = (1, 512)
KOKORO_SPEAKER_EMB_SHAPE = (510, 1, 256)
KOKORO_SAMPLE_RATE = 24000
QWEN3_OMNI_SAMPLE_RATE = 24000
QWEN3_OMNI_DEFAULT_SPEAKER = "Ethan"
LOGGER = logging.getLogger(__name__)


class _Speech:
    def __init__(self, data):
        self.data = data


class _SpeechResult:
    def __init__(self, data, sample_rate: int, text: str = ""):
        self.speeches = [_Speech(data)]
        self.output_sample_rate = sample_rate
        self.text = text


SPEAKER_SCORE_COL = "speaker score"
CONTENT_SCORE_COL = "content score"
DURATION_SCORE_COL = "duration score"
ACOUSTIC_SCORE_COL = "acoustic score"
TEXT_WER_COL = "text WER"
TEXT_SIM_COL = "text similarity"
OVERALL_SCORE_COL = "overall similarity"
TEXT_COL = "generated_text"


def _qwen3_omni_speakers(source: Any) -> list[str]:
    """Return supported Qwen3-Omni speaker names from a transformers config, dict, or model dir."""
    if source is None:
        return []
    if isinstance(source, (str, Path)):
        config_path = Path(source) / "config.json"
        if not config_path.exists():
            return []
        try:
            import json

            with config_path.open("r", encoding="utf-8") as config_file:
                source = json.load(config_file)
        except Exception:
            return []

    talker = source.get("talker_config") if isinstance(source, dict) else getattr(source, "talker_config", None)
    if talker is None:
        return []
    speaker_id = talker.get("speaker_id") if isinstance(talker, dict) else getattr(talker, "speaker_id", None)
    return list(speaker_id.keys()) if isinstance(speaker_id, dict) else []


class _Qwen3OmniSpeakerMixin:
    """Speaker resolution logic shared by Qwen3-Omni HF and GenAI wrappers."""

    model_type = "speech-generation"
    prompts_file = OMNI_PROMPTS_FILE

    def _init_speakers(self, supported_speakers: list[str], requested_default: str) -> None:
        self.supported_speakers = supported_speakers
        # Case-insensitive lookup: user-supplied "ethan" resolves to the model's "Ethan".
        self._speaker_map = {s.strip().lower(): s for s in supported_speakers if isinstance(s, str) and s.strip()}
        self.default_speaker = self._pick_default(requested_default)

    def _pick_default(self, requested: str) -> str:
        requested = requested.strip() if isinstance(requested, str) else ""
        if requested.lower() in self._speaker_map:
            return self._speaker_map[requested.lower()]
        if not self.supported_speakers:
            return requested or QWEN3_OMNI_DEFAULT_SPEAKER
        fallback = sorted(self.supported_speakers, key=str.lower)[0]
        if requested:
            LOGGER.info(
                "Qwen3-Omni speaker '%s' is unavailable for this model. Falling back to '%s'.",
                requested,
                fallback,
            )
        return fallback

    def get_speaker_embedding_shape(self) -> tuple:
        return (1, 1)

    def _resolve_speaker(self, voice: str) -> str:
        speaker = voice.strip() if isinstance(voice, str) and voice.strip() else self.default_speaker
        if not self._speaker_map:
            return speaker
        resolved = self._speaker_map.get(speaker.lower())
        if resolved is None:
            supported = ", ".join(sorted(self.supported_speakers, key=str.lower))
            raise ValueError(
                f"Unsupported Qwen3-Omni speaker '{speaker}'. Supported speakers for this model: {supported}."
            )
        return resolved

    @staticmethod
    def _reject_unsupported_inputs(speaker_embedding: Any, language: str) -> None:
        if speaker_embedding is not None:
            raise ValueError(
                "Qwen3-Omni selects the voice by a named speaker and does not accept speaker embeddings. "
                "Use --speech-voice with a supported speaker name."
            )
        if isinstance(language, str) and language.strip():
            raise ValueError(
                "Qwen3-Omni does not support language selection via --speech-language "
                "(it is currently supported only for Kokoro). Remove --speech-language."
            )


class SpeechT5Wrapper:
    """Wrapper for SpeechT5 models (HF/Optimum)."""

    def __init__(self, model, processor, vocoder):
        self.model = model
        self.processor = processor
        self.vocoder = vocoder
        self.model_type = "speech-generation"

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.model, attr)

    def get_speaker_embedding_shape(self):
        # SpeechT5 expects a single xvector with 512 values.
        return SPEECHT5_SPEAKER_EMB_SHAPE

    @staticmethod
    def resolve_default_speaker_embedding_file() -> str:
        from huggingface_hub import hf_hub_download

        embedding_file = hf_hub_download(
            repo_id=DEFAULT_SPEAKER_EMBEDDING_REPO_ID,
            filename=DEFAULT_SPEAKER_EMBEDDING_FILENAME,
            repo_type="dataset",
        )
        LOGGER.info(
            "Using default speaker embeddings for SpeechT5: %s/%s -> %s",
            DEFAULT_SPEAKER_EMBEDDING_REPO_ID,
            DEFAULT_SPEAKER_EMBEDDING_FILENAME,
            embedding_file,
        )
        return embedding_file

    def generate(self, prompt, speaker_embedding=None, **_kwargs):
        if speaker_embedding is None:
            raise ValueError(
                "This model requires speaker embeddings but none were provided. "
                "Pass --speaker_embeddings with a .bin or .npy float32 xvector file."
            )

        import torch

        input_data = self.processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
        if hasattr(input_data, "pop"):
            input_data.pop("token_type_ids", None)

        input_tokens = (
            input_data["input_ids"] if hasattr(input_data, "keys") and "input_ids" in input_data else input_data
        )
        if isinstance(input_tokens, (list, tuple)):
            input_tokens = torch.tensor(input_tokens, dtype=torch.long)
        elif not isinstance(input_tokens, torch.Tensor):
            input_tokens = torch.as_tensor(input_tokens)
        if input_tokens.ndim == 1:
            input_tokens = input_tokens.unsqueeze(0)

        emb = torch.as_tensor(
            speaker_embedding.data if hasattr(speaker_embedding, "data") else speaker_embedding,
            dtype=torch.float32,
        )
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)

        generation_kwargs = {"speaker_embeddings": emb}
        if self.vocoder is not None:
            generation_kwargs["vocoder"] = self.vocoder

        with torch.inference_mode():
            output = self.model.generate(input_tokens, **generation_kwargs)
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, torch.Tensor):
                speech = output.detach().cpu().reshape(-1).numpy()
            else:
                speech = torch.as_tensor(output).cpu().reshape(-1).numpy()

        return _SpeechResult(speech, 16000)


class KokoroModelWrapper:
    """Unified wrapper for Kokoro (HF or Optimum) via KPipeline.

    HF path uses KModel + KPipeline.
    Optimum path uses OVModelForTextToSpeechSeq2Seq preprocess_input() + generate().
    """

    def __init__(self, model_id, ov_model=None):
        from kokoro import KPipeline
        from kokoro.model import KModel

        self.model_type = "speech-generation"
        self._model_id = str(model_id)
        self._lang_code = "a"
        self._ov_model = ov_model

        if ov_model is not None:
            self._kmodel = None
            self._pipeline = None
        else:
            model_dir = Path(self._model_id)
            if model_dir.is_dir():
                config_path = model_dir / "config.json"
                if not config_path.exists():
                    raise ValueError(f"Kokoro local model directory is missing config.json: {config_path}")
                self._kmodel = KModel(config=str(config_path))
            else:
                # if model_id is not a local directory, KModel will fetch config using repo_id.
                self._kmodel = KModel(repo_id=self._model_id)

            self._pipeline = KPipeline(lang_code=self._lang_code, model=self._kmodel)

    @staticmethod
    def _normalize_kokoro_lang_code(language: str) -> str:
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

    def get_speaker_embedding_shape(self):
        return KOKORO_SPEAKER_EMB_SHAPE

    def generate(self, prompt, speaker_embedding=None, language="", voice="", **_kwargs):
        requested_lang_code = self._normalize_kokoro_lang_code(language)
        selected_voice = voice.strip() if isinstance(voice, str) else ""
        if not selected_voice:
            selected_voice = "af_heart"

        # if optimum path
        if self._ov_model is not None:
            preprocess_kwargs = {
                "text": prompt,
                "lang_code": requested_lang_code,
            }
            if speaker_embedding is not None:
                preprocess_kwargs["speaker_embedding"] = (
                    speaker_embedding.data if hasattr(speaker_embedding, "data") else speaker_embedding
                )
            else:
                preprocess_kwargs["voice"] = selected_voice

            preprocessed = self._ov_model.preprocess_input(**preprocess_kwargs)
            output = self._ov_model.generate(**preprocessed)

            try:
                import torch

                if isinstance(output, torch.Tensor):
                    audio = output.detach().cpu().reshape(-1).numpy()
                else:
                    audio = torch.as_tensor(output).cpu().reshape(-1).numpy()
            except ImportError:
                audio = np.asarray(output, dtype=np.float32).reshape(-1)

            return _SpeechResult(audio, KOKORO_SAMPLE_RATE)

        if requested_lang_code != self._lang_code:
            self._lang_code = requested_lang_code
            from kokoro import KPipeline

            # Recreate pipeline with new language using the same underlying KModel.
            self._pipeline = KPipeline(lang_code=self._lang_code, model=self._kmodel)

        generator = self._pipeline(prompt, voice=selected_voice)
        result = next(iter(generator), None)
        if result is None:
            raise ValueError("Kokoro pipeline returned no audio output.")

        audio = result.audio

        return _SpeechResult(audio, KOKORO_SAMPLE_RATE)


def _seed_deterministic_generation():
    import torch

    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Qwen3OmniSpeechWrapper(_Qwen3OmniSpeakerMixin):
    """Wrapper for Qwen3-Omni HF (transformers) or Optimum models that emit speech via the talker module."""

    def __init__(self, model, processor, default_speaker=QWEN3_OMNI_DEFAULT_SPEAKER):
        self.model = model
        self.processor = processor
        self._dtypes_aligned = False
        self._init_speakers(_qwen3_omni_speakers(getattr(model, "config", None)), default_speaker)

    def __getattr__(self, attr):
        return getattr(self.model, attr)

    def _align_hf_talker_dtype_if_needed(self):
        """Reconcile talker/thinker dtypes on the underlying HF model.

        The dense Qwen3-Omni checkpoint ships the thinker and talker in different precisions,
        which breaks the CPU matmul between them. We cast the talker's text_projection to the
        thinker's dtype and monkey-patch `_get_talker_user_parts` to cast the intermediate
        hidden states, so the two modules can interoperate. Runs at most once per wrapper.
        """
        if self._dtypes_aligned:
            return
        self._dtypes_aligned = True

        try:
            import types
            import torch

            thinker_dtype = self.model.thinker.get_input_embeddings().weight.dtype
            talker = self.model.talker
            projection = talker.text_projection
            projection_params = list(projection.parameters())
            projection_dtype = projection_params[0].dtype if projection_params else thinker_dtype

            if projection_dtype != thinker_dtype:
                talker.text_projection = projection.to(dtype=thinker_dtype)
                LOGGER.info(
                    "Aligned Qwen3-Omni talker text_projection dtype from %s to %s.",
                    projection_dtype,
                    thinker_dtype,
                )

            if talker.dtype == thinker_dtype or getattr(self.model, "_wwb_talker_dtype_patch_applied", False):
                return

            def _patched_get_talker_user_parts(
                model_self, im_start_index, segment_end_index, multimodal_mask, thinker_hidden, thinker_embed
            ):
                talker = model_self.talker
                user_talker_part = torch.empty(
                    (1, segment_end_index - im_start_index, model_self.config.talker_config.text_config.hidden_size),
                    device=talker.device,
                    dtype=talker.dtype,
                )
                user_mm_mask = multimodal_mask[:, im_start_index:segment_end_index]

                if user_mm_mask.any():
                    mm_input = thinker_hidden[:, im_start_index:segment_end_index][user_mm_mask]
                    user_talker_part[user_mm_mask] = talker.hidden_projection(mm_input).to(
                        talker.device, dtype=user_talker_part.dtype
                    )

                text_input = thinker_embed[:, im_start_index:segment_end_index][~user_mm_mask]
                user_talker_part[~user_mm_mask] = talker.text_projection(text_input).to(
                    talker.device, dtype=user_talker_part.dtype
                )
                return user_talker_part

            self.model._get_talker_user_parts = types.MethodType(_patched_get_talker_user_parts, self.model)
            self.model._wwb_talker_dtype_patch_applied = True
            LOGGER.info(
                "Applied WWB Qwen3-Omni CPU dtype compatibility patch (talker=%s, thinker=%s).",
                talker.dtype,
                thinker_dtype,
            )
        except Exception:
            # Model internals differ between versions; leave the model unmodified on any mismatch.
            pass

    def generate(self, prompt, speaker_embedding=None, language="", voice="", **_kwargs):
        self._reject_unsupported_inputs(speaker_embedding, language)

        import torch

        _seed_deterministic_generation()

        speaker = self._resolve_speaker(voice)
        conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Optimum OVModelForMultimodalLM keeps CPU torch tensors and has no torch device attribute;
        # HF models expose one and require inputs to live on it.
        device = getattr(self.model, "device", None)
        if isinstance(device, torch.device):
            inputs = inputs.to(device)

        with torch.inference_mode():
            self._align_hf_talker_dtype_if_needed()

            # HF rejects temperature=0 but honors do_sample=False; Optimum ignores do_sample.
            if type(self.model).__module__.startswith("transformers"):
                greedy_kwargs = {"thinker_do_sample": False, "talker_do_sample": False}
            else:
                greedy_kwargs = {"thinker_temperature": 0, "talker_temperature": 0}
            output = self.model.generate(
                **inputs,
                speaker=speaker,
                return_audio=True,
                thinker_max_new_tokens=128,
                talker_max_new_tokens=128,
                thinker_eos_token_id=151645,
                **greedy_kwargs,
            )

        if isinstance(output, (tuple, list)):
            thinker_sequences, audio = output[0], output[1]
        else:
            thinker_sequences = getattr(output, "sequences", None)
            audio = getattr(output, "audio", None)
        if audio is None:
            raise ValueError("Qwen3-Omni did not return audio. Ensure the talker module is enabled.")

        speech = torch.as_tensor(audio).detach().cpu().reshape(-1).float().numpy()

        # Decode only the newly generated tokens (drop the prompt prefix). The talker's audio is
        # conditioned step-by-step on these tokens' hidden states, so this transcript matches the
        # audio and can be compared to the reference transcript.
        text = ""
        if thinker_sequences is not None:
            input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
            prompt_len = input_ids.shape[-1] if isinstance(input_ids, torch.Tensor) else 0
            generated_ids = thinker_sequences[0, prompt_len:] if prompt_len else thinker_sequences[0]
            text = self.processor.batch_decode(
                [generated_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        return _SpeechResult(speech, QWEN3_OMNI_SAMPLE_RATE, text=text)


class GenAIOmniSpeechWrapper(_Qwen3OmniSpeakerMixin):
    """Adapts openvino_genai.OmniPipeline to the speech-generation evaluator interface."""

    def __init__(self, pipe, model_dir, default_speaker=QWEN3_OMNI_DEFAULT_SPEAKER):
        self.pipe = pipe
        self.model_dir = model_dir
        self._init_speakers(_qwen3_omni_speakers(model_dir), default_speaker)

    def generate(self, prompt, speaker_embedding=None, language="", voice="", **_kwargs):
        self._reject_unsupported_inputs(speaker_embedding, language)

        import openvino_genai

        _seed_deterministic_generation()

        text_config = self.pipe.get_vlm().get_generation_config()
        text_config.do_sample = False
        text_config.max_new_tokens = 128
        text_config.eos_token_id = 151645

        talker_speech_config = openvino_genai.OmniTalkerSpeechConfig()
        talker_speech_config.return_audio = True
        talker_speech_config.speaker = self._resolve_speaker(voice)
        talker_speech_config.max_new_tokens = 128
        talker_speech_config.rng_seed = 42
        talker_speech_config.talker_top_k = 1
        talker_speech_config.cp_top_k = 1

        result = self.pipe.generate(
            prompt,
            text_config=text_config,
            talker_speech_config=talker_speech_config,
        )

        speech_outputs = getattr(result.speech_result, "waveforms", None)
        if speech_outputs is None:
            speech_outputs = getattr(result.speech_result, "speech_outputs", None)
        if not speech_outputs:
            raise ValueError("OmniPipeline did not return audio. Ensure the talker module is enabled.")

        speech = np.asarray(speech_outputs[0].data, dtype=np.float32).reshape(-1)
        texts = getattr(result, "texts", None) or []
        return _SpeechResult(speech, QWEN3_OMNI_SAMPLE_RATE, text=texts[0] if texts else "")


def _safe_metric_mean(values):
    arr = np.array([np.nan if value is None else value for value in values], dtype=float)
    if np.isnan(arr).all():
        return np.nan
    return np.nanmean(arr)


@register_evaluator("speech-generation")
class SpeechGenerationEvaluator(BaseEvaluator):
    def __init__(
        self,
        base_model: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        num_samples: int = None,
        gen_speech_fn=None,
        speaker_embedding_file_path: str = None,
        whisper_model: str = "base.en",
        vocoder_path: str = None,
        speech_language: str = "",
        speech_voice: str = "",
    ) -> None:
        if base_model is None and gt_data is None:
            raise ValueError("Speech generation pipeline for evaluation or ground truth data must be defined")

        self.test_data = test_data
        self.num_samples = num_samples
        self.generation_fn = gen_speech_fn
        self.whisper_model = whisper_model
        self.vocoder_path = vocoder_path
        self.last_cmp = None
        self.speaker_embedding_file_path = speaker_embedding_file_path
        self.speaker_embedding = None
        self.speech_language = speech_language.strip().lower() if isinstance(speech_language, str) else ""
        self.speech_voice = speech_voice.strip() if isinstance(speech_voice, str) else ""

        if self.speaker_embedding_file_path is not None and not os.path.exists(self.speaker_embedding_file_path):
            raise ValueError(f"Speaker embedding file does not exist: {self.speaker_embedding_file_path}")
        # Speaker embedding tensor shape depends on backend (SpeechT5 vs Kokoro), so load lazily.
        self.speaker_embedding = None

        self.gt_dir = os.path.dirname(gt_data) if gt_data else os.getcwd()

        self._evaluator = TTSSimilarityEvaluator(
            whisper_model=self.whisper_model,
        )

        if base_model:
            self.gt_data = self._generate_data(base_model, gen_speech_fn, os.path.join(self.gt_dir, "reference"))
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)

        self._validate_required_columns(self.gt_data, ["audio", "prompts"], "ground truth data")

    def get_generation_fn(self):
        return self.generation_fn

    def score(self, model_or_data, gen_speech_fn=None, output_dir=None, verbose=False, **kwargs):
        if verbose:
            LOGGER.setLevel(logging.DEBUG)
        audio_folder = os.path.join(output_dir if output_dir else self.gt_dir, "target")

        if isinstance(model_or_data, str) and os.path.exists(model_or_data):
            predictions = pd.read_csv(model_or_data, keep_default_na=False)
        else:
            predictions = self._generate_data(model_or_data, gen_speech_fn, audio_folder)

        self._validate_required_columns(predictions, ["audio", "prompts"], "prediction data")
        self.predictions = predictions

        has_text = TEXT_COL in self.gt_data.columns and TEXT_COL in predictions.columns
        max_samples = min(len(self.gt_data), len(predictions))

        # Per-metric list of per-prompt values, in the column order used in the returned DataFrame.
        per_prompt: dict[str, list] = {
            col: []
            for col in (
                SPEAKER_SCORE_COL,
                CONTENT_SCORE_COL,
                DURATION_SCORE_COL,
                ACOUSTIC_SCORE_COL,
                TEXT_WER_COL,
                TEXT_SIM_COL,
                OVERALL_SCORE_COL,
            )
        }

        for idx in tqdm(range(max_samples), desc="TTS similarity evaluation"):
            gt_row = self.gt_data.iloc[idx]
            prediction_row = predictions.iloc[idx]

            scores = self._evaluator.evaluate(
                target_path=str(prediction_row["audio"]),
                reference_path=str(gt_row["audio"]),
                verbose=verbose,
            )
            per_prompt[SPEAKER_SCORE_COL].append(scores.speaker)
            per_prompt[CONTENT_SCORE_COL].append(scores.content)
            per_prompt[DURATION_SCORE_COL].append(scores.duration)
            per_prompt[ACOUSTIC_SCORE_COL].append(scores.acoustic)
            per_prompt[OVERALL_SCORE_COL].append(scores.overall)

            if has_text:
                wer, sim = self._score_text_pair(str(gt_row[TEXT_COL]), str(prediction_row[TEXT_COL]))
            else:
                wer, sim = None, None
            per_prompt[TEXT_WER_COL].append(wer)
            per_prompt[TEXT_SIM_COL].append(sim)

            if verbose and has_text:
                LOGGER.debug("--- Thinker Text ---")
                LOGGER.debug("Reference text: %s", str(gt_row[TEXT_COL]))
                LOGGER.debug("Target text:    %s", str(prediction_row[TEXT_COL]))
                LOGGER.debug("Text WER: %s", "None" if wer is None else f"{wer:.3f}")
                LOGGER.debug("Text similarity: %s", "None" if sim is None else f"{sim:.3f}")

        aggregated = {col: _safe_metric_mean(values) for col, values in per_prompt.items()}

        gt_texts = (
            self.gt_data[TEXT_COL].values[:max_samples] if TEXT_COL in self.gt_data.columns else [""] * max_samples
        )
        pred_texts = (
            predictions[TEXT_COL].values[:max_samples] if TEXT_COL in predictions.columns else [""] * max_samples
        )
        self.last_cmp = pd.DataFrame(
            {
                **per_prompt,
                "prompt": predictions["prompts"].values[:max_samples],
                "source_model": self.gt_data["audio"].values[:max_samples],
                "optimized_model": predictions["audio"].values[:max_samples],
                "reference_text": gt_texts,
                "target_text": pred_texts,
            }
        )

        return pd.DataFrame(per_prompt), pd.DataFrame([aggregated])

    @staticmethod
    def _score_text_pair(reference: str, target: str):
        """Return (WER, similarity in 0..1) — reuses TTSSimilarityEvaluator's normalize + content-score curve
        so text and audio WER-based scores sit on the same scale side-by-side."""
        from jiwer import wer as jiwer_wer

        ref = normalize_text(reference or "")
        tgt = normalize_text(target or "")
        if not ref and not tgt:
            return 0.0, 1.0
        if not ref:
            return None, None
        wer = safe_float(jiwer_wer(ref, tgt))
        return wer, linear_distance_score(wer, 0.0, ScoringConfig().content_wer_bad)

    def worst_examples(self, top_k: int = 5, metric=OVERALL_SCORE_COL):
        assert self.last_cmp is not None

        res = self.last_cmp.nsmallest(top_k, metric)
        return [row for _, row in res.iterrows()]

    def _validate_required_columns(self, data: pd.DataFrame, required_columns: list[str], data_name: str) -> None:
        missing_columns = [column for column in required_columns if column not in data.columns]
        if missing_columns:
            raise ValueError(f"{data_name.capitalize()} is missing required columns: {', '.join(missing_columns)}")

    def _load_speaker_embedding(self, speaker_embedding_file_path: str, expected_shape=None):
        if speaker_embedding_file_path is None:
            return None

        import openvino as ov

        embedding_path = Path(speaker_embedding_file_path)
        if embedding_path.suffix.lower() == ".npy":
            speaker_embedding = np.load(embedding_path)
        else:
            speaker_embedding = np.fromfile(embedding_path, dtype=np.float32)

        speaker_embedding = np.asarray(speaker_embedding, dtype=np.float32).reshape(-1)
        if speaker_embedding.size == 0:
            raise ValueError(f"Speaker embedding file is empty: {speaker_embedding_file_path}")

        if expected_shape is None:
            expected_shape = SPEECHT5_SPEAKER_EMB_SHAPE

        expected_dims = tuple(int(dim) for dim in expected_shape)
        expected_flat_size = int(np.prod(expected_dims))
        if speaker_embedding.size != expected_flat_size:
            raise ValueError(
                f"Unexpected speaker embedding size {speaker_embedding.size} in {speaker_embedding_file_path}. "
                f"Expected flattened size {expected_flat_size} for shape {expected_dims}."
            )
        return ov.Tensor(speaker_embedding.reshape(expected_dims))

    def _ensure_default_speaker_embedding_if_needed(self, model) -> None:
        """Lazily load default speaker embedding for backends that require one.

        SpeechT5 requires explicit speaker embeddings. Kokoro handles voice/style
        selection in its own wrapper, so no default embedding is prepared here.
        """
        if self.speaker_embedding is not None or self.speaker_embedding_file_path is not None:
            return

        if hasattr(model, "resolve_default_speaker_embedding_file"):
            self.speaker_embedding_file_path = model.resolve_default_speaker_embedding_file()
            expected_shape = tuple(int(dim) for dim in model.get_speaker_embedding_shape())
            self.speaker_embedding = self._load_speaker_embedding(self.speaker_embedding_file_path, expected_shape)

    def _generate_data(self, model, gen_speech_fn=None, audio_dir="reference"):
        def default_gen_speech_fn(model, prompt, speaker_embedding=None, language="", voice=""):
            result = model.generate(prompt, speaker_embedding, language=language, voice=voice)
            audio_data = np.array(result.speeches[0].data).reshape(-1)
            try:
                sr = int(result.output_sample_rate)
            except AttributeError:
                sr = 16000
            text = getattr(result, "text", "") or ""
            return audio_data, sr, text

        generation_fn = gen_speech_fn or default_gen_speech_fn

        if hasattr(model, "_reject_unsupported_inputs"):
            model._reject_unsupported_inputs(self.speaker_embedding_file_path, self.speech_language)

        self._ensure_default_speaker_embedding_if_needed(model)

        if self.test_data:
            if isinstance(self.test_data, str):
                data = pd.read_csv(self.test_data)
            elif isinstance(self.test_data, dict):
                data = pd.DataFrame.from_dict(dict(self.test_data))
            else:
                data = pd.DataFrame.from_dict({"prompts": list(self.test_data)})
        else:
            prompts_file = getattr(model, "prompts_file", None) or PROMPTS_FILE
            data_path = files("whowhatbench.prompts").joinpath(prompts_file)
            prompt_data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
            data = pd.DataFrame.from_dict(prompt_data["en"])

        self._validate_required_columns(data, ["prompts"], "input prompt data")
        if self.num_samples is not None:
            data = data.iloc[: self.num_samples]

        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)

        audios = []
        texts = []
        prompt_values = data["prompts"].values
        expected_shape = tuple(int(dim) for dim in model.get_speaker_embedding_shape())

        for idx, prompt in tqdm(enumerate(prompt_values), total=len(prompt_values), desc="Evaluate pipeline"):
            speaker_embedding_file_path = self.speaker_embedding_file_path
            if "speaker_embeddings" in data.columns and pd.notna(data.iloc[idx]["speaker_embeddings"]):
                speaker_embedding_file_path = data.iloc[idx]["speaker_embeddings"]
                if isinstance(speaker_embedding_file_path, str) and speaker_embedding_file_path.strip() == "":
                    speaker_embedding_file_path = None

            if speaker_embedding_file_path:
                speaker_embedding = self._load_speaker_embedding(speaker_embedding_file_path, expected_shape)
            else:
                speaker_embedding = self.speaker_embedding

            result = generation_fn(
                model,
                prompt,
                speaker_embedding=speaker_embedding,
                language=self.speech_language,
                voice=self.speech_voice,
            )
            # Custom gen_speech_fn may still return the legacy (audio, sr) tuple; accept both.
            if len(result) == 3:
                generated_audio, generated_sr, generated_text = result
            else:
                generated_audio, generated_sr = result
                generated_text = ""

            audio_path = os.path.join(audio_dir, f"{idx}.wav")
            sf.write(audio_path, generated_audio, samplerate=generated_sr)
            audios.append(audio_path)
            texts.append(generated_text or "")

        return pd.DataFrame(
            {
                "prompts": list(prompt_values),
                "audio": audios,
                TEXT_COL: texts,
            }
        )
