# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Union

import os
import pandas as pd
from tqdm import tqdm

from .registry import register_evaluator, BaseEvaluator
from .utils import no_double_bos
from .whowhat_metrics import WordErrorRate

DEFAULT_ASR_INSTRUCTION = "Transcribe this audio."


@register_evaluator("speech-recognition")
class SpeechRecognitionEvaluator(BaseEvaluator):
    def __init__(
        self,
        base_model: Any = None,
        gt_data: str = None,
        test_data: Union[str, dict] = None,
        processor: Any = None,
        max_new_tokens: int = 256,
        num_samples: int = None,
        gen_answer_fn=None,
        instruction: str = DEFAULT_ASR_INSTRUCTION,
    ) -> None:
        if base_model is None and gt_data is None:
            raise ValueError("Speech recognition pipeline for evaluation or ground truth data must be defined")

        self.test_data = test_data
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.num_samples = num_samples
        self.instruction = instruction
        self.generation_fn = gen_answer_fn
        self.wer = WordErrorRate()
        self.last_cmp = None

        if base_model:
            self.gt_data = self._generate_data(base_model, gen_answer_fn)
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)

    def get_generation_fn(self):
        return self.generation_fn

    def _transcribe(self, model, audio):
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": self.instruction},
                ],
            }
        ]
        with no_double_bos(self.processor):
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        device = getattr(model, "device", None)
        if isinstance(device, torch.device):
            inputs = inputs.to(device)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(tokens[:, input_len:], skip_special_tokens=True)[0]

    def _generate_data(self, model, gen_answer_fn=None):
        gen_answer_fn = gen_answer_fn or self._transcribe

        if not isinstance(self.test_data, dict):
            raise ValueError("Speech recognition requires audio test data (provide --dataset).")
        data = pd.DataFrame.from_dict(self.test_data)
        if self.num_samples is not None:
            data = data.iloc[: self.num_samples]

        answers = [gen_answer_fn(model, audio) for audio in tqdm(data["audio"].values, desc="Evaluate pipeline")]
        return pd.DataFrame({"prompts": list(data["prompts"].values), "answers": answers})

    def score(self, model_or_data, gen_answer_fn=None, output_dir=None, verbose=False, **kwargs):
        if isinstance(model_or_data, str) and os.path.exists(model_or_data):
            predictions = pd.read_csv(model_or_data, keep_default_na=False)
        else:
            predictions = self._generate_data(model_or_data, gen_answer_fn)
        self.predictions = predictions

        self._validate_columns(self.gt_data, "Ground truth")
        self._validate_columns(predictions, "Prediction")
        if len(self.gt_data) != len(predictions):
            raise ValueError(
                f"Ground truth ({len(self.gt_data)} rows) and predictions ({len(predictions)} rows) differ in length"
            )
        if not (self.gt_data["prompts"].values == predictions["prompts"].values).all():
            raise ValueError("Ground truth and prediction audio ids ('prompts') do not match")

        metric_dict, per_prompt = self.wer.evaluate(self.gt_data, predictions)
        self.last_cmp = pd.DataFrame(
            {
                "prompt": self.gt_data["prompts"].values,
                "source_model": self.gt_data["answers"].values,
                "optimized_model": predictions["answers"].values,
                "WER": per_prompt["WER"],
            }
        )
        return pd.DataFrame(per_prompt), pd.DataFrame([metric_dict])

    @staticmethod
    def _validate_columns(data, name):
        for column in ("prompts", "answers"):
            if column not in data.columns:
                raise ValueError(f"{name} data is missing required column '{column}'")

    def worst_examples(self, top_k: int = 5, metric="WER"):
        assert self.last_cmp is not None
        res = self.last_cmp.nlargest(top_k, metric)
        return [row for _, row in res.iterrows()]
