# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Union

import pandas as pd
from tqdm import tqdm

from .registry import register_evaluator, BaseEvaluator
from .whowhat_metrics import TranscriptSimilarity


@register_evaluator("speech-recognition")
class SpeechRecognitionEvaluator(BaseEvaluator):
    def __init__(
        self,
        base_model: Any = None,
        gt_data: str = None,
        test_data: Union[str, dict] = None,
        max_new_tokens: int = 256,
        num_samples: int = None,
        gen_answer_fn=None,
        speech_language: str = "",
    ) -> None:
        if base_model is None and gt_data is None:
            raise ValueError("Speech recognition pipeline for evaluation or ground truth data must be defined")

        self.test_data = test_data
        self.max_new_tokens = max_new_tokens
        self.num_samples = num_samples
        self.generation_fn = gen_answer_fn
        self.similarity = TranscriptSimilarity(speech_language)
        self.last_cmp = None

        if base_model:
            self.gt_data = self._generate_data(base_model, gen_answer_fn)
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)
            if num_samples is not None:
                self.gt_data = self.gt_data.iloc[:num_samples]

    def get_generation_fn(self):
        return self.generation_fn

    def _generate_data(self, model, gen_answer_fn=None):
        if not isinstance(self.test_data, dict):
            raise ValueError("Speech recognition requires audio test data (provide --dataset).")
        data = pd.DataFrame.from_dict(self.test_data)
        if self.num_samples is not None:
            data = data.iloc[: self.num_samples]

        if gen_answer_fn is None:
            answers = [
                model.transcribe(audio, self.max_new_tokens)
                for audio in tqdm(data["audio"].values, desc="Evaluate pipeline")
            ]
        else:
            answers = [
                gen_answer_fn(model, audio, self.max_new_tokens)
                for audio in tqdm(data["audio"].values, desc="Evaluate pipeline")
            ]
        return pd.DataFrame({"prompts": list(data["prompts"].values), "answers": answers})

    def score(self, model_or_data, gen_answer_fn=None, output_dir=None, verbose=False, **kwargs):
        if isinstance(model_or_data, str) and os.path.exists(model_or_data):
            predictions = pd.read_csv(model_or_data, keep_default_na=False)
        else:
            predictions = self._generate_data(model_or_data, gen_answer_fn)
        if self.num_samples is not None:
            predictions = predictions.iloc[: self.num_samples]
        elif len(self.gt_data) != len(predictions):
            raise ValueError(
                f"Ground truth ({len(self.gt_data)} rows) and predictions ({len(predictions)} rows) differ in length"
            )
        self.predictions = predictions

        self._validate_columns(self.gt_data, "Ground truth")
        self._validate_columns(predictions, "Prediction")
        if not (self.gt_data["prompts"].astype(str).values == predictions["prompts"].astype(str).values).all():
            raise ValueError("Ground truth and prediction audio ids ('prompts') do not match")

        metric, per_prompt_metric = self.similarity.evaluate(self.gt_data, predictions)
        self.last_cmp = pd.DataFrame(
            {
                "prompt": self.gt_data["prompts"].values,
                "source_model": self.gt_data["answers"].values,
                "optimized_model": predictions["answers"].values,
                "similarity": per_prompt_metric["similarity"],
            }
        )
        return pd.DataFrame(per_prompt_metric), pd.DataFrame([metric])

    @staticmethod
    def _validate_columns(data, name):
        for column in ("prompts", "answers"):
            if column not in data.columns:
                raise ValueError(f"{name} data is missing required column '{column}'")

    def worst_examples(self, top_k: int = 5, metric="similarity"):
        assert self.last_cmp is not None
        res = self.last_cmp.nsmallest(top_k, metric)
        return [row for _, row in res.iterrows()]
