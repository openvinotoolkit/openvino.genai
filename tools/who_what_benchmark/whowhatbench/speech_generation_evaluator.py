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
from .tts_similarity import TTSSimilarityEvaluator

PROMPTS_FILE = "speech_generation_prompts.yaml"
DEFAULT_SPEAKER_EMBEDDING_REPO_ID = "Xenova/cmu-arctic-xvectors-extracted"
DEFAULT_SPEAKER_EMBEDDING_FILENAME = "cmu_us_slt_arctic-wav-arctic_a0508.bin"
SPEECHT5_SPEAKER_EMB_SHAPE = (1, 512)
KOKORO_SPEAKER_EMB_SHAPE = (510, 1, 256)
KOKORO_SAMPLE_RATE = 24000
LOGGER = logging.getLogger(__name__)

SPEAKER_SCORE_COL = "speaker score"
CONTENT_SCORE_COL = "content score"
DURATION_SCORE_COL = "duration score"
ACOUSTIC_SCORE_COL = "acoustic score"
OVERALL_SCORE_COL = "overall similarity"


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

        class _Speech:
            def __init__(self, data):
                self.data = data

        class _SpeechResult:
            def __init__(self, data):
                self.speeches = [_Speech(data)]
                self.output_sample_rate = 16000

        return _SpeechResult(speech)


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

        class _Speech:
            def __init__(self, data):
                self.data = data

        class _SpeechResult:
            def __init__(self, data):
                self.speeches = [_Speech(data)]
                self.output_sample_rate = KOKORO_SAMPLE_RATE

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

            return _SpeechResult(audio)

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

        return _SpeechResult(audio)


class Qwen3CustomVoiceWrapper:
    """Unified wrapper for Qwen3 CustomVoice models via HF or GenAI backends."""

    def __init__(self, model):
        self.model = model
        self.model_type = "speech-generation"

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.model, attr)

    def get_speaker_embedding_shape(self):
        return None

    @staticmethod
    def _preview_ids(values, max_items=80):
        seq = list(values)
        head = seq[:max_items]
        suffix = ",..." if len(seq) > max_items else ""
        return f"len={len(seq)} [{','.join(str(int(x)) for x in head)}{suffix}]"

    def generate(self, prompt, speaker_embedding=None, language="", voice="", instruct="", **kwargs):
        if speaker_embedding is not None:
            LOGGER.debug("Ignoring speaker_embedding for Qwen3 CustomVoice.")

        selected_speaker = voice.strip() if isinstance(voice, str) else ""
        if not selected_speaker:
            raise ValueError("Qwen3 CustomVoice requires --speech-voice to select a speaker.")

        selected_language = language.strip() if isinstance(language, str) else ""
        selected_instruct = instruct.strip() if isinstance(instruct, str) else ""

        # Keep WWB speech comparisons deterministic for Qwen3 unless explicitly overridden.
        kwargs.setdefault("do_sample", False)
        kwargs.setdefault("subtalker_dosample", False)

        if os.getenv("WWB_QWEN3_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
            LOGGER.info(
                "[WWB_QWEN3_DEBUG] speaker='%s' language='%s' instruct_len=%d do_sample=%s subtalker_dosample=%s",
                selected_speaker,
                selected_language,
                len(selected_instruct),
                kwargs.get("do_sample"),
                kwargs.get("subtalker_dosample"),
            )

            if hasattr(self.model, "processor"):
                try:
                    assistant_text = f"<|im_start|>assistant\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                    assistant_tok = self.model.processor(text=assistant_text, return_tensors="pt", padding=True)
                    assistant_ids = assistant_tok["input_ids"].detach().cpu().reshape(-1).tolist()
                    LOGGER.info("[WWB_QWEN3_DEBUG] assistant_text=%r", assistant_text)
                    LOGGER.info("[WWB_QWEN3_DEBUG] assistant_input_ids %s", self._preview_ids(assistant_ids))

                    if selected_instruct:
                        instruct_text = f"<|im_start|>user\n{selected_instruct}<|im_end|>\n"
                        instruct_tok = self.model.processor(text=instruct_text, return_tensors="pt", padding=True)
                        instruct_ids = instruct_tok["input_ids"].detach().cpu().reshape(-1).tolist()
                        LOGGER.info("[WWB_QWEN3_DEBUG] instruct_text=%r", instruct_text)
                        LOGGER.info("[WWB_QWEN3_DEBUG] instruct_ids %s", self._preview_ids(instruct_ids))
                except Exception as exc:
                    LOGGER.warning("[WWB_QWEN3_DEBUG] failed to tokenize debug prompt: %s", exc)

        if hasattr(self.model, "generate_custom_voice"):
            wavs, sample_rate = self.model.generate_custom_voice(
                text=prompt,
                speaker=selected_speaker,
                language=selected_language or "Auto",
                instruct=selected_instruct,
                **kwargs,
            )

            class _Speech:
                def __init__(self, data):
                    self.data = data

            class _SpeechResult:
                def __init__(self, data, output_sample_rate):
                    self.speeches = [_Speech(data)]
                    self.output_sample_rate = output_sample_rate

            return _SpeechResult(np.array(wavs[0]).reshape(-1), sample_rate)

        generation_properties = {"speaker": selected_speaker}
        if selected_language:
            generation_properties["language"] = selected_language
        if selected_instruct:
            generation_properties["instruct"] = selected_instruct

        generation_properties.update(kwargs)

        return self.model.generate(prompt, **generation_properties)


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
        speech_instruct: str = "",
        max_new_tokens: int = None,
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
        self.speech_language = speech_language.strip() if isinstance(speech_language, str) else ""
        self.speech_voice = speech_voice.strip() if isinstance(speech_voice, str) else ""
        self.speech_instruct = speech_instruct.strip() if isinstance(speech_instruct, str) else ""
        self.max_new_tokens = max_new_tokens

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
        audio_folder = os.path.join(output_dir if output_dir else self.gt_dir, "target")

        if isinstance(model_or_data, str) and os.path.exists(model_or_data):
            predictions = pd.read_csv(model_or_data, keep_default_na=False)
        else:
            predictions = self._generate_data(model_or_data, gen_speech_fn, audio_folder)

        self._validate_required_columns(predictions, ["audio", "prompts"], "prediction data")
        self.predictions = predictions

        max_samples = min(len(self.gt_data), len(predictions))
        speaker_scores = []
        content_scores = []
        duration_scores = []
        acoustic_scores = []
        overall_scores = []

        for idx in tqdm(range(max_samples), desc="TTS similarity evaluation"):
            gt_row = self.gt_data.iloc[idx]
            prediction_row = predictions.iloc[idx]

            scores = self._evaluator.evaluate(
                target_path=str(prediction_row["audio"]),
                reference_path=str(gt_row["audio"]),
                verbose=verbose,
            )

            speaker_scores.append(scores.speaker)
            content_scores.append(scores.content)
            duration_scores.append(scores.duration)
            acoustic_scores.append(scores.acoustic)
            overall_scores.append(scores.overall)

        all_metrics_per_prompt = {
            SPEAKER_SCORE_COL: speaker_scores,
            CONTENT_SCORE_COL: content_scores,
            DURATION_SCORE_COL: duration_scores,
            ACOUSTIC_SCORE_COL: acoustic_scores,
            OVERALL_SCORE_COL: overall_scores,
        }

        all_metrics = {
            SPEAKER_SCORE_COL: _safe_metric_mean(speaker_scores),
            CONTENT_SCORE_COL: _safe_metric_mean(content_scores),
            DURATION_SCORE_COL: _safe_metric_mean(duration_scores),
            ACOUSTIC_SCORE_COL: _safe_metric_mean(acoustic_scores),
            OVERALL_SCORE_COL: _safe_metric_mean(overall_scores),
        }

        self.last_cmp = pd.DataFrame(
            {
                **all_metrics_per_prompt,
                "prompt": predictions["prompts"].values[:max_samples],
                "source_model": self.gt_data["audio"].values[:max_samples],
                "optimized_model": predictions["audio"].values[:max_samples],
            }
        )

        return pd.DataFrame(all_metrics_per_prompt), pd.DataFrame([all_metrics])

    def worst_examples(self, top_k: int = 5, metric=OVERALL_SCORE_COL):
        assert self.last_cmp is not None

        res = self.last_cmp.nsmallest(top_k, metric)
        return [row for _, row in res.iterrows()]

    def _validate_required_columns(self, data: pd.DataFrame, required_columns: list[str], data_name: str) -> None:
        missing_columns = [column for column in required_columns if column not in data.columns]
        if missing_columns:
            raise ValueError(f"{data_name.capitalize()} is missing required columns: {', '.join(missing_columns)}")

    @staticmethod
    def _get_expected_speaker_embedding_shape(model):
        getter = getattr(model, "get_speaker_embedding_shape", None)
        if not callable(getter):
            return None

        shape = getter()
        if shape is None:
            return None
        return tuple(int(dim) for dim in shape)

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

        expected_shape = self._get_expected_speaker_embedding_shape(model)
        if expected_shape is None:
            return

        if hasattr(model, "resolve_default_speaker_embedding_file"):
            self.speaker_embedding_file_path = model.resolve_default_speaker_embedding_file()
            self.speaker_embedding = self._load_speaker_embedding(self.speaker_embedding_file_path, expected_shape)

    def _generate_data(self, model, gen_speech_fn=None, audio_dir="reference"):
        def default_gen_speech_fn(
            model,
            prompt,
            speaker_embedding=None,
            language="",
            voice="",
            instruct="",
            max_new_tokens=None,
        ):
            generation_kwargs = {}
            effective_max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
            if effective_max_new_tokens is not None:
                generation_kwargs["max_new_tokens"] = effective_max_new_tokens
            result = model.generate(
                prompt,
                speaker_embedding,
                language=language,
                voice=voice,
                instruct=instruct,
                **generation_kwargs,
            )
            audio_data = np.array(result.speeches[0].data).reshape(-1)
            try:
                sr = int(result.output_sample_rate)
            except AttributeError:
                sr = 16000
            return audio_data, sr

        generation_fn = gen_speech_fn or default_gen_speech_fn

        self._ensure_default_speaker_embedding_if_needed(model)

        if self.test_data:
            if isinstance(self.test_data, str):
                data = pd.read_csv(self.test_data)
            elif isinstance(self.test_data, dict):
                data = pd.DataFrame.from_dict(dict(self.test_data))
            else:
                data = pd.DataFrame.from_dict({"prompts": list(self.test_data)})
        else:
            data_path = files("whowhatbench.prompts").joinpath(PROMPTS_FILE)
            prompt_data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
            data = pd.DataFrame.from_dict(prompt_data["en"])

        self._validate_required_columns(data, ["prompts"], "input prompt data")
        if self.num_samples is not None:
            data = data.iloc[: self.num_samples]

        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)

        audios = []
        prompt_values = data["prompts"].values
        expected_shape = self._get_expected_speaker_embedding_shape(model)

        for idx, prompt in tqdm(enumerate(prompt_values), total=len(prompt_values), desc="Evaluate pipeline"):
            speaker_embedding_file_path = self.speaker_embedding_file_path
            if "speaker_embeddings" in data.columns and pd.notna(data.iloc[idx]["speaker_embeddings"]):
                speaker_embedding_file_path = data.iloc[idx]["speaker_embeddings"]
                if isinstance(speaker_embedding_file_path, str) and speaker_embedding_file_path.strip() == "":
                    speaker_embedding_file_path = None

            if expected_shape is not None and speaker_embedding_file_path:
                speaker_embedding = self._load_speaker_embedding(speaker_embedding_file_path, expected_shape)
            else:
                speaker_embedding = self.speaker_embedding

            speech_language = self.speech_language
            if "speech_language" in data.columns and pd.notna(data.iloc[idx]["speech_language"]):
                speech_language = str(data.iloc[idx]["speech_language"]).strip()

            speech_voice = self.speech_voice
            if "speech_voice" in data.columns and pd.notna(data.iloc[idx]["speech_voice"]):
                speech_voice = str(data.iloc[idx]["speech_voice"]).strip()

            speech_instruct = self.speech_instruct
            if "speech_instruct" in data.columns and pd.notna(data.iloc[idx]["speech_instruct"]):
                speech_instruct = str(data.iloc[idx]["speech_instruct"]).strip()

            generated_audio, generated_sr = generation_fn(
                model,
                prompt,
                speaker_embedding=speaker_embedding,
                language=speech_language,
                voice=speech_voice,
                instruct=speech_instruct,
                max_new_tokens=self.max_new_tokens,
            )

            audio_path = os.path.join(audio_dir, f"{idx}.wav")
            sf.write(audio_path, generated_audio, samplerate=generated_sr)
            audios.append(audio_path)

        return pd.DataFrame(
            {
                "prompts": list(prompt_values),
                "audio": audios,
            }
        )
