# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import logging
import os
from typing import Any, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from .registry import register_evaluator, BaseEvaluator
from .utils import (
    no_double_bos,
    AUDIO_SAMPLING_RATE,
    get_model_type,
)
from .whowhat_metrics import TranscriptSimilarity

logger = logging.getLogger(__name__)

FUNASR_TOKENIZER_SUBFOLDER = (
    "Qwen3-0.6B"  # Source layout: https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/tree/main/Qwen3-0.6B
)
ASR_MODEL_TYPES = {"funasr", "fun_asr"}
AUDIO_VLM_MODEL_TYPES = {"gemma4", "gemma4_unified"}

DEFAULT_ASR_INSTRUCTION = "Transcribe this audio."
# Language specific prompt https://huggingface.co/google/gemma-4-12B#6-audio
DEFAULT_ASR_INSTRUCTION_WITH_LANGUAGE = (
    "Transcribe the following speech segment in {language} into {language} text.\n\n"
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, "
    "and write 3 instead of three."
)


def asr_instruction(language: str = "") -> str:
    return DEFAULT_ASR_INSTRUCTION_WITH_LANGUAGE.format(language=language) if language else DEFAULT_ASR_INSTRUCTION


@contextlib.contextmanager
def _silenced_output():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield


class MultimodalTranscriber:
    def __init__(self, model: Any, model_id: str, language: str = "") -> None:
        from transformers import AutoProcessor

        self.model = model
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=False)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.instruction = asr_instruction(language)

    def transcribe(self, audio, max_new_tokens: int) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.instruction},
                    {"type": "audio", "audio": audio},
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
                trust_remote_code=True,
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


class FunASROptimumTranscriber:
    def __init__(self, model: Any, tokenizer: Any, language: str = "") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess_kwargs = {"language": language} if language else {}

    def transcribe(self, audio, max_new_tokens: int) -> str:
        import torch

        inputs = self.model.preprocess_input(audio, sampling_rate=AUDIO_SAMPLING_RATE, **self.preprocess_kwargs)
        prompt_len = inputs["decoder_input_ids"].shape[-1]
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
            generation_kwargs["language"] = self.language
        result = self.pipeline.generate(np.asarray(audio, dtype=np.float32).tolist(), **generation_kwargs)
        return result.texts[0].strip()


def _get_asr_model_type(model_id: str) -> str:
    model_type = get_model_type(model_id)
    if model_type is None:
        raise ValueError(f"Cannot determine the speech recognition model type for '{model_id}'")
    return model_type


def _load_multimodal_model(model_id, device, ov_config, use_hf, use_genai, **kwargs):
    from .model_loaders import load_visual_text_model

    kwargs["model_type"] = "visual-text"
    return load_visual_text_model(model_id, device, ov_config, use_hf, use_genai, **kwargs)


class ASRHFTranscriber:
    @staticmethod
    def create(model_id, device="CPU", ov_config=None, language="", **kwargs):
        if _get_asr_model_type(model_id) in ASR_MODEL_TYPES:
            logger.info("Using FunASR API")
            return FunASRSourceTranscriber(model_id, language or "en")

        model = _load_multimodal_model(model_id, device, ov_config, True, False, **kwargs)
        return MultimodalTranscriber(model, model_id, language or "English")


class ASRGenAITranscriber:
    @staticmethod
    def create(model_id, device="CPU", ov_config=None, language="", **kwargs):
        model_type = _get_asr_model_type(model_id)
        if model_type in ASR_MODEL_TYPES:
            logger.info("Using OpenVINO GenAI ASRPipeline API")
            import openvino_genai

            pipeline = openvino_genai.ASRPipeline(str(model_id), device.upper(), **(ov_config or {}))
            return FunASRGenAITranscriber(pipeline, language or "en")
        if model_type in AUDIO_VLM_MODEL_TYPES:
            raise ValueError("Gemma4 audio input is not yet supported by the OpenVINO GenAI backend")

        model = _load_multimodal_model(model_id, device, ov_config, False, True, **kwargs)
        return GenAIMultimodalTranscriber(model, language or "English")


class ASROptimumTranscriber:
    @staticmethod
    def create(model_id, device="CPU", ov_config=None, language="", **kwargs):
        model_type = _get_asr_model_type(model_id)
        if model_type in ASR_MODEL_TYPES:
            logger.info("Using Optimum API")
            from optimum.intel.openvino import OVModelForSpeechSeq2Seq
            from transformers import AutoTokenizer

            model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, device=device, ov_config=ov_config)
            subfolder = FUNASR_TOKENIZER_SUBFOLDER if model_type == "funasr" else ""
            tokenizer = AutoTokenizer.from_pretrained(str(model_id), subfolder=subfolder)
            return FunASROptimumTranscriber(model, tokenizer, language or "en")

        model = _load_multimodal_model(model_id, device, ov_config, False, False, **kwargs)
        return MultimodalTranscriber(model, model_id, language or "English")


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
