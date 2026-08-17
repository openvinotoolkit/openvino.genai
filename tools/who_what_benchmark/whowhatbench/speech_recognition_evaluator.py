# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Union

import os
import pandas as pd
from tqdm import tqdm

from .registry import register_evaluator, BaseEvaluator
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
        tokenizer: Any = None,
        max_new_tokens: int = 256,
        num_samples: int = None,
        gen_answer_fn=None,
        instruction: str = DEFAULT_ASR_INSTRUCTION,
        device: str = "CPU",
    ) -> None:
        if base_model is None and gt_data is None:
            raise ValueError("Speech recognition pipeline for evaluation or ground truth data must be defined")

        self.test_data = test_data
        self.processor = processor
        self.tokenizer = tokenizer
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

    def _transcribe(self, model, audio, sampling_rate):
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

        # Gemma chat templates already emit the bos token; avoid duplicating it.
        tokenizer = getattr(self.processor, "tokenizer", None)
        orig_add_bos_token = getattr(tokenizer, "add_bos_token", None)
        if (
            orig_add_bos_token is not None
            and getattr(tokenizer, "chat_template", None)
            and "bos_token" in tokenizer.chat_template
        ):
            tokenizer.add_bos_token = False
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            if orig_add_bos_token is not None:
                tokenizer.add_bos_token = orig_add_bos_token

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

        answers = []
        for audio, sampling_rate in tqdm(
            zip(data["audio"].values, data["sampling_rate"].values), total=len(data), desc="Evaluate pipeline"
        ):
            answers.append(gen_answer_fn(model, audio, int(sampling_rate)))

        return pd.DataFrame({"prompts": list(data["prompts"].values), "answers": answers})

    def score(self, model_or_data, gen_answer_fn=None, output_dir=None, verbose=False, **kwargs):
        if isinstance(model_or_data, str) and os.path.exists(model_or_data):
            predictions = pd.read_csv(model_or_data, keep_default_na=False)
        else:
            predictions = self._generate_data(model_or_data, gen_answer_fn)
        self.predictions = predictions

        metric_dict, per_prompt = self.wer.evaluate(self.gt_data, predictions)

        compared = min(len(self.gt_data), len(predictions))
        self.last_cmp = pd.DataFrame(
            {
                "prompt": self.gt_data["prompts"].values[:compared],
                "source_model": self.gt_data["answers"].values[:compared],
                "optimized_model": predictions["answers"].values[:compared],
                "WER": per_prompt["WER"][:compared],
            }
        )

        return pd.DataFrame(per_prompt), pd.DataFrame([metric_dict])

    def worst_examples(self, top_k: int = 5, metric="WER"):
        assert self.last_cmp is not None
        res = self.last_cmp.nlargest(top_k, metric)
        return [row for _, row in res.iterrows()]
