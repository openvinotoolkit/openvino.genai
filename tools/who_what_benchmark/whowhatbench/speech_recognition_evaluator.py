# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import os
from typing import Any, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from .registry import register_evaluator, BaseEvaluator
from .utils import no_double_bos
from .whowhat_metrics import WordErrorRate

AUDIO_SAMPLING_RATE = 16000

DEFAULT_ASR_INSTRUCTION = "Transcribe this audio."


def asr_instruction(language: str = "") -> str:
    return f"Transcribe this audio in {language}." if language else DEFAULT_ASR_INSTRUCTION


@contextlib.contextmanager
def _silenced_output():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield


class MultimodalTranscriber:
    def __init__(self, model: Any, processor: Any, language: str = "") -> None:
        self.model = model
        self.processor = processor
        self.instruction = asr_instruction(language)

    def transcribe(self, audio, max_new_tokens: int) -> str:
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
        device = getattr(self.model, "device", None)
        if isinstance(device, torch.device):
            inputs = inputs.to(device)

        prompt_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(tokens[:, prompt_len:], skip_special_tokens=True)[0]


class GenAIMultimodalTranscriber:
    def __init__(self, pipeline: Any, language: str = "") -> None:
        self.pipeline = pipeline
        self.instruction = asr_instruction(language)

    def transcribe(self, audio, max_new_tokens: int) -> str:
        import openvino as ov

        result = self.pipeline.generate(
            self.instruction,
            audios=[ov.Tensor(np.asarray(audio, dtype=np.float32))],
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
        return result.texts[0]


class FunASRSourceTranscriber:
    def __init__(self, model_id: str, language: str = "") -> None:
        try:
            from funasr import AutoModel
        except ImportError as error:
            raise ModuleNotFoundError(
                "The `funasr` package is required to evaluate FunASR source models with --hf. "
                "Please install it with `pip install funasr`."
            ) from error

        self.language = language or None
        with _silenced_output():
            self.model = AutoModel(
                model=str(model_id),
                hub="hf",
                device="cpu",
                disable_update=True,
            )

    def transcribe(self, audio, max_new_tokens: int) -> str:
        import torch

        with _silenced_output():
            results = self.model.generate(
                input=[torch.as_tensor(np.asarray(audio, dtype=np.float32))],
                cache={},
                batch_size=1,
                language=self.language,  # None keeps the language-neutral funasr prompt
                itn=True,
                max_length=max_new_tokens,
            )
        return str(results[0]["text"]).strip()


class OVDetokenizer:
    def __init__(self, detokenizer_path) -> None:
        import openvino_tokenizers  # noqa: F401 - registers the tokenizer extension in openvino
        from openvino import Core

        self.detokenizer = Core().compile_model(str(detokenizer_path), "CPU")

    def batch_decode(self, ids, **kwargs):
        return [str(text) for text in self.detokenizer(np.asarray(ids, dtype=np.int64))[0]]


class FunASROptimumTranscriber:
    def __init__(self, model: Any, tokenizer: Any, language: str = "") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess_kwargs = {"language": language} if language else {}

    def transcribe(self, audio, max_new_tokens: int) -> str:
        import torch

        inputs = self.model.preprocess_input(audio, sampling_rate=AUDIO_SAMPLING_RATE, **self.preprocess_kwargs)
        prompt_len = inputs["decoder_input_ids"].shape[1]
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        tokens = getattr(tokens, "sequences", tokens)
        # generate() returns the prompt followed by the generated ids, decode the generated part only.
        return self.tokenizer.batch_decode(tokens[:, prompt_len:], skip_special_tokens=True)[0].strip()


class FunASRGenAITranscriber:
    def __init__(self, pipeline: Any, language: str = "") -> None:
        self.pipeline = pipeline
        self.language = language

    def transcribe(self, audio, max_new_tokens: int) -> str:
        generation_kwargs = {"max_new_tokens": max_new_tokens}
        if self.language:
            # Omitting the argument keeps the prompt language-neutral, an empty string would be
            # treated as a forced (empty) language.
            generation_kwargs["language"] = self.language
        result = self.pipeline.generate(np.asarray(audio, dtype=np.float32).tolist(), **generation_kwargs)
        return result.texts[0].strip()


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
    ) -> None:
        if base_model is None and gt_data is None:
            raise ValueError("Speech recognition pipeline for evaluation or ground truth data must be defined")

        self.test_data = test_data
        self.max_new_tokens = max_new_tokens
        self.num_samples = num_samples
        self.generation_fn = gen_answer_fn
        self.wer = WordErrorRate()
        self.last_cmp = None

        if base_model:
            self.gt_data = self._generate_data(base_model, gen_answer_fn)
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)

    def get_generation_fn(self):
        return self.generation_fn

    @staticmethod
    def _transcribe(model, audio, max_new_tokens):
        transcribe = getattr(model, "transcribe", None)
        if transcribe is None:
            raise TypeError(
                f"{type(model).__name__} does not provide 'transcribe(audio, max_new_tokens)'. Speech recognition "
                "expects a transcriber returned by load_speech_recognition_model(), or an explicit gen_answer_fn."
            )
        return transcribe(audio, max_new_tokens)

    def _generate_data(self, model, gen_answer_fn=None):
        gen_answer_fn = gen_answer_fn or self._transcribe

        if not isinstance(self.test_data, dict):
            raise ValueError("Speech recognition requires audio test data (provide --dataset).")
        data = pd.DataFrame.from_dict(self.test_data)
        if self.num_samples is not None:
            data = data.iloc[: self.num_samples]

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
        self.predictions = predictions

        self._validate_columns(self.gt_data, "Ground truth")
        self._validate_columns(predictions, "Prediction")
        if len(self.gt_data) != len(predictions):
            raise ValueError(
                f"Ground truth ({len(self.gt_data)} rows) and predictions ({len(predictions)} rows) differ in length"
            )
        if not (self.gt_data["prompts"].astype(str).values == predictions["prompts"].astype(str).values).all():
            raise ValueError("Ground truth and prediction audio ids ('prompts') do not match")

        wer, per_prompt_wer = self.wer.evaluate(self.gt_data, predictions)
        similarity = max(0.0, 1.0 - wer["WER"])
        per_prompt_similarity = [max(0.0, 1.0 - value) for value in per_prompt_wer["WER"]]
        self.last_cmp = pd.DataFrame(
            {
                "prompt": self.gt_data["prompts"].values,
                "source_model": self.gt_data["answers"].values,
                "optimized_model": predictions["answers"].values,
                "similarity": per_prompt_similarity,
            }
        )
        return pd.DataFrame({"similarity": per_prompt_similarity}), pd.DataFrame([{"similarity": similarity}])

    @staticmethod
    def _validate_columns(data, name):
        for column in ("prompts", "answers"):
            if column not in data.columns:
                raise ValueError(f"{name} data is missing required column '{column}'")

    def worst_examples(self, top_k: int = 5, metric="similarity"):
        assert self.last_cmp is not None
        res = self.last_cmp.nsmallest(top_k, metric)
        return [row for _, row in res.iterrows()]
