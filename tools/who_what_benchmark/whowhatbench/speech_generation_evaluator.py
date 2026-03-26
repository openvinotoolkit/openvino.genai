# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Union

import os

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from importlib.resources import files
from tqdm import tqdm

from .registry import register_evaluator, BaseEvaluator
from .tts_similarity import TTSSimilarityEvaluator

PROMPTS_FILE = "speech_generation_prompts.yaml"

SPEAKER_SCORE_COL = "speaker score"
CONTENT_SCORE_COL = "content score"
DURATION_SCORE_COL = "duration score"
OVERALL_SCORE_COL = "overall score"


class TextToSpeechModelWrapper:
    """Wrapper for non-GenAI speech generation models (HF/Optimum) to provide evaluator-compatible interface."""

    def __init__(self, model, processor, vocoder):
        self.model = model
        self.processor = processor
        self.vocoder = vocoder
        self.model_type = "speech-generation"

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.model, attr)

    def generate(self, prompt, speaker_embedding=None, **_kwargs):
        if speaker_embedding is None:
            raise ValueError(
                "This model requires speaker embeddings but none were provided. "
                "Pass --speaker_embeddings with a binary float32 xvector file."
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
        sample_rate: int = 16000,
        speaker_embedding_file_path: str = None,
        whisper_model: str = "small",
    ) -> None:
        assert base_model is not None or gt_data is not None, (
            "Speech generation pipeline for evaluation or ground truth data must be defined"
        )

        self.test_data = test_data
        self.num_samples = num_samples
        self.generation_fn = gen_speech_fn
        self.sample_rate = sample_rate
        self.whisper_model = whisper_model
        self.whisper_device = "cpu"
        self.whisper_compute_type = "default"
        self.last_cmp = None
        self.speaker_embedding_file_path = speaker_embedding_file_path
        self.speaker_embedding = None

        if self.speaker_embedding_file_path is not None and not os.path.exists(self.speaker_embedding_file_path):
            raise ValueError(f"Speaker embedding file does not exist: {self.speaker_embedding_file_path}")
        self.speaker_embedding = self._load_speaker_embedding(self.speaker_embedding_file_path)

        self.gt_dir = os.path.dirname(gt_data) if gt_data else os.getcwd()

        self._evaluator = TTSSimilarityEvaluator(
            sample_rate=self.sample_rate,
            whisper_model=self.whisper_model,
            whisper_device=self.whisper_device,
            whisper_compute_type=self.whisper_compute_type,
        )

        if base_model:
            self.gt_data = self._generate_data(base_model, gen_speech_fn, os.path.join(self.gt_dir, "reference"))
        else:
            self.gt_data = self._normalize_input_data(pd.read_csv(gt_data, keep_default_na=False))

        self._validate_required_columns(self.gt_data, ["audio"], "ground truth data")

    def get_generation_fn(self):
        return self.generation_fn

    def score(self, model_or_data, gen_speech_fn=None, output_dir=None, verbose=False, **kwargs):
        audio_folder = os.path.join(output_dir if output_dir else self.gt_dir, "target")

        if isinstance(model_or_data, str) and os.path.exists(model_or_data):
            predictions = self._normalize_input_data(pd.read_csv(model_or_data, keep_default_na=False))
        else:
            predictions = self._generate_data(model_or_data, gen_speech_fn, audio_folder)

        self._validate_required_columns(predictions, ["audio", "prompts"], "prediction data")
        self.predictions = predictions

        max_samples = min(len(self.gt_data), len(predictions))
        speaker_scores = []
        content_scores = []
        duration_scores = []
        overall_scores = []

        for idx in tqdm(range(max_samples), desc="TTS similarity evaluation"):
            gt_row = self.gt_data.iloc[idx]
            prediction_row = predictions.iloc[idx]

            prompt = prediction_row.get("prompts", gt_row.get("prompts", ""))
            expected_text = prediction_row.get("expected_text", gt_row.get("expected_text", prompt))
            if pd.isna(expected_text):
                expected_text = prompt
            expected_text = str(expected_text)
            language = prediction_row.get("language", gt_row.get("language", None))
            if pd.isna(language):
                language = None
            elif isinstance(language, str) and language.strip() == "":
                language = None
            elif language is not None:
                language = str(language)

            scores = self._evaluator.evaluate(
                target_path=str(prediction_row["audio"]),
                reference_path=str(gt_row["audio"]),
                expected_text=expected_text,
                language=language,
                verbose=verbose,
            )

            speaker_scores.append(scores.speaker)
            content_scores.append(scores.content)
            duration_scores.append(scores.duration)
            overall_scores.append(scores.overall)

        all_metrics_per_prompt = {
            SPEAKER_SCORE_COL: speaker_scores,
            CONTENT_SCORE_COL: content_scores,
            DURATION_SCORE_COL: duration_scores,
            OVERALL_SCORE_COL: overall_scores,
        }

        all_metrics = {
            SPEAKER_SCORE_COL: _safe_metric_mean(speaker_scores),
            CONTENT_SCORE_COL: _safe_metric_mean(content_scores),
            DURATION_SCORE_COL: _safe_metric_mean(duration_scores),
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

    def _normalize_input_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if "prompts" not in data.columns and "prompt" in data.columns:
            data = data.rename(columns={"prompt": "prompts"})

        if "prompts" not in data.columns:
            raise ValueError("Input prompt data must contain 'prompts' or 'prompt' column")

        if "expected_text" not in data.columns:
            data["expected_text"] = data["prompts"]

        return data

    def _validate_required_columns(self, data: pd.DataFrame, required_columns: list[str], data_name: str) -> None:
        missing_columns = [column for column in required_columns if column not in data.columns]
        if missing_columns:
            raise ValueError(f"{data_name.capitalize()} is missing required columns: {', '.join(missing_columns)}")

    def _load_speaker_embedding(self, speaker_embedding_file_path: str):
        if speaker_embedding_file_path is None:
            return None

        import openvino as ov

        speaker_embedding = np.fromfile(speaker_embedding_file_path, dtype=np.float32)
        if speaker_embedding.size == 0:
            raise ValueError(f"Speaker embedding file is empty: {speaker_embedding_file_path}")
        if speaker_embedding.size != 512:
            raise ValueError(
                f"Unexpected speaker embedding size {speaker_embedding.size} in {speaker_embedding_file_path}. "
                "Expected flattened size 512."
            )
        return ov.Tensor(speaker_embedding.reshape(1, 512))

    def _generate_data(self, model, gen_speech_fn=None, audio_dir="reference"):

        def default_gen_speech_fn(model, prompt, speaker_embedding=None, voice="", language=""):
            generation_properties = {}
            if voice:
                generation_properties["voice"] = voice
            if language:
                generation_properties["language"] = language

            result = model.generate(prompt, speaker_embedding, **generation_properties)
            audio_data = np.array(result.speeches[0].data).reshape(-1)
            try:
                sr = int(result.output_sample_rate)
            except AttributeError:
                sr = 16000
            return audio_data, sr

        generation_fn = gen_speech_fn or default_gen_speech_fn

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

        data = self._normalize_input_data(data)
        if self.num_samples is not None:
            data = data.iloc[: self.num_samples]

        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)

        audios = []
        prompt_values = data["prompts"].values
        expected_text_values = data["expected_text"].values
        language_values = data["language"].values if "language" in data.columns else [None] * len(data)
        voice_values = data["voice"].values if "voice" in data.columns else [""] * len(data)

        for idx, prompt in tqdm(enumerate(prompt_values), total=len(prompt_values), desc="Evaluate pipeline"):
            speaker_embedding_file_path = self.speaker_embedding_file_path
            if "speaker_embeddings" in data.columns and pd.notna(data.iloc[idx]["speaker_embeddings"]):
                speaker_embedding_file_path = data.iloc[idx]["speaker_embeddings"]
                if isinstance(speaker_embedding_file_path, str) and speaker_embedding_file_path.strip() == "":
                    speaker_embedding_file_path = None

            if speaker_embedding_file_path:
                speaker_embedding = self._load_speaker_embedding(speaker_embedding_file_path)
            else:
                speaker_embedding = self.speaker_embedding

            generated_audio, generated_sr = generation_fn(
                model,
                prompt,
                speaker_embedding=speaker_embedding,
                voice=str(voice_values[idx]) if pd.notna(voice_values[idx]) else "",
            )

            audio_path = os.path.join(audio_dir, f"{idx}.wav")
            if generated_sr != self.sample_rate:
                import librosa

                generated_audio = librosa.resample(
                    generated_audio.astype(float), orig_sr=generated_sr, target_sr=self.sample_rate
                )
            sf.write(audio_path, generated_audio, samplerate=self.sample_rate)
            audios.append(audio_path)

        return pd.DataFrame(
            {
                "prompts": list(prompt_values),
                "expected_text": list(expected_text_values),
                "language": [str(item) if pd.notna(item) else "" for item in language_values],
                "voice": [str(item) if pd.notna(item) else "" for item in voice_values],
                "audio": audios,
            }
        )
