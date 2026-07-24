# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
import time


class Qwen3ASROptimumPipeline:
    SAMPLE_RATE = 16000
    EOS_TOKEN_IDS = [151643, 151645]

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def preprocess(self, sample, **kwargs):
        start = time.perf_counter()
        generate_kwargs = kwargs.get("generate_kwargs", {})
        language = generate_kwargs.get("language") or kwargs.get("language")

        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if language:
            text_prompt += f"language {language}<asr_text>"

        inputs = self.processor(text=text_prompt, audio=sample, sampling_rate=self.SAMPLE_RATE, return_tensors="pt")
        end = time.perf_counter()
        return inputs, end - start, language

    def generate(self, sample, **kwargs):
        max_new_tokens = kwargs.get("max_new_tokens", 1000)
        generate_kwargs = kwargs.get("generate_kwargs", {})
        max_new_tokens = generate_kwargs.get("max_new_tokens", max_new_tokens)

        inputs, preprocess_time, language = self.preprocess(sample, **kwargs)

        start_gen = time.perf_counter()
        output_ids = self.model.generate(
            input_features=inputs["input_features"],
            decoder_input_ids=inputs["input_ids"],
            eos_token_id=self.EOS_TOKEN_IDS,
            max_new_tokens=max_new_tokens,
        )
        end_gen = time.perf_counter()
        generation_time = end_gen - start_gen

        start_detok = time.perf_counter()
        prompt_len = inputs["input_ids"].shape[1]
        generated_only = output_ids[:, prompt_len:]
        full_text = self.processor.batch_decode(generated_only, skip_special_tokens=False)[0]
        parsed_output = self.parse_asr_output(full_text)
        end_detok = time.perf_counter()
        detokenization_time = end_detok - start_detok

        return {
            "text": parsed_output["text"],
            "language": language if language else parsed_output["language"],
            "perf_metrics": {
                "preprocess_time": preprocess_time * 1000,
                "generation_time": generation_time,
                "detokenization_time": detokenization_time * 1000,
            },
        }

    def parse_asr_output(self, raw_text):
        language_match = re.search(r"<\|([a-z]{2,3})\|>", raw_text)
        text_match = re.search(r"<asr_text>(.*?)(?:<\||$)", raw_text.replace("<|asr_text|>", "<asr_text>"))

        return {
            "language": language_match.group(1) if language_match else None,
            "text": text_match.group(1).strip() if text_match else raw_text.strip(),
        }

    def __call__(self, sample, **kwargs):
        return self.generate(sample, **kwargs)

    @staticmethod
    def init_model(model_type):
        if model_type == "qwen3-asr":
            from qwen_asr import Qwen3ASRModel  # noqa: F401
            from qwen_asr.core.transformers_backend import Qwen3ASRProcessor  # noqa: F401
