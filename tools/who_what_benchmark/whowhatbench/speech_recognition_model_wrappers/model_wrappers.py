# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import logging
from typing import Any

import numpy as np

from ..utils import (
    no_double_bos,
    AUDIO_SAMPLING_RATE,
    get_model_type,
)

logger = logging.getLogger(__name__)

FUNASR_TOKENIZER_SUBFOLDER = (
    "Qwen3-0.6B"  # Source layout: https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/tree/main/Qwen3-0.6B
)
FUNASR_MODEL_TYPES = {"funasr", "fun_asr"}
WHISPER_MODEL_TYPES = {"whisper"}

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
    def __init__(self, model_id: str, language: str = "", torch_dtype=None) -> None:
        import torch

        try:
            from funasr import AutoModel
        except ImportError as error:
            raise ModuleNotFoundError(
                "The `funasr` package is required to evaluate FunASR source models with --hf. "
                "Please install it with `pip install funasr`."
            ) from error

        self.language = language or None
        precision_kwargs = {}
        if torch_dtype is not None:
            precision_kwargs = {
                "fp16": False,
                "bf16": False,
                "llm_dtype": {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[torch_dtype],
            }

        with _silenced_output():
            self.model = AutoModel(
                model=str(model_id),
                hub="hf",
                device="cpu",
                disable_update=True,
                trust_remote_code=True,
                **precision_kwargs,
            )

        if torch_dtype is not None:
            # Cast only the decoder: casting the audio encoder breaks inference with its FP32 inputs.
            self.model.model.llm.to(dtype=torch_dtype)

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
    def __init__(self, model_id, model_type, device, ov_config, language: str = "") -> None:
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq
        from transformers import AutoTokenizer

        self.model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, device=device, ov_config=ov_config)
        subfolder = FUNASR_TOKENIZER_SUBFOLDER if model_type == "funasr" else ""
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_id), subfolder=subfolder)
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


class GenAISpeechRecognitionTranscriber:
    def __init__(self, model_id, device, ov_config, language: str = "") -> None:
        import openvino_genai

        self.pipeline = openvino_genai.ASRPipeline(str(model_id), device.upper(), **(ov_config or {}))
        self.language = language

    def transcribe(self, audio, max_new_tokens: int) -> str:
        generation_kwargs = {"max_new_tokens": max_new_tokens}
        if self.language:
            generation_kwargs["language"] = self.language
        result = self.pipeline.generate(np.asarray(audio, dtype=np.float32).tolist(), **generation_kwargs)
        return result.texts[0].strip()


def _transcribe_whisper(model, processor, audio, language, max_new_tokens, device=None, dtype=None) -> str:
    import torch

    inputs = processor(
        np.asarray(audio, dtype=np.float32),
        sampling_rate=AUDIO_SAMPLING_RATE,
        return_tensors="pt",
        truncation=False,
        return_attention_mask=True,
    )
    input_features = inputs.input_features
    attention_mask = inputs.attention_mask
    if device is not None:
        input_features = input_features.to(device=device, dtype=dtype)
        attention_mask = attention_mask.to(device=device)
    # English-only checkpoints reject both `task` and `language`, and their language is always English
    is_multilingual = getattr(model.generation_config, "is_multilingual", True)
    if not is_multilingual and language:
        logger.warning("Whisper model is English-only, ignoring language '%s'", language)
    with torch.inference_mode():
        tokens = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            return_timestamps=input_features.shape[-1] > 2 * model.config.max_source_positions,
            language=language if is_multilingual else None,
            task="transcribe" if is_multilingual else None,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


class WhisperHFTranscriber:
    def __init__(self, model_id: str, language: str = "", torch_dtype=None) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.processor = WhisperProcessor.from_pretrained(str(model_id))
        model_kwargs = {"torch_dtype": torch_dtype} if torch_dtype is not None else {}
        self.model = WhisperForConditionalGeneration.from_pretrained(str(model_id), **model_kwargs).eval()
        # None lets Whisper detect the language from the audio
        self.language = language or None

    def transcribe(self, audio, max_new_tokens: int) -> str:
        return _transcribe_whisper(
            self.model,
            self.processor,
            audio,
            self.language,
            max_new_tokens,
            device=self.model.device,
            dtype=self.model.dtype,
        )


class WhisperOptimumTranscriber:
    def __init__(self, model_id, device, ov_config, language: str = "") -> None:
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq
        from transformers import WhisperProcessor

        self.model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, device=device, ov_config=ov_config)
        self.processor = WhisperProcessor.from_pretrained(str(model_id))
        # None lets Whisper detect the language from the audio
        self.language = language or None

    def transcribe(self, audio, max_new_tokens: int) -> str:
        return _transcribe_whisper(self.model, self.processor, audio, self.language, max_new_tokens)


def _get_asr_model_type(model_id: str) -> str:
    model_type = get_model_type(model_id)
    if model_type is None:
        raise ValueError(f"Cannot determine the speech recognition model type for '{model_id}'")
    return model_type


def _load_multimodal_model(model_id, device, ov_config, use_hf, use_genai, **kwargs):
    from ..model_loaders import load_visual_text_model

    kwargs["model_type"] = "visual-text"
    return load_visual_text_model(model_id, device, ov_config, use_hf, use_genai, **kwargs)


class ASRHFTranscriber:
    @staticmethod
    def create(model_id, device="CPU", ov_config=None, language="", **kwargs):
        model_type = _get_asr_model_type(model_id)
        if model_type in FUNASR_MODEL_TYPES:
            logger.info("Using FunASR API")
            return FunASRSourceTranscriber(
                model_id,
                language or "en",
                torch_dtype=kwargs.get("torch_dtype"),
            )
        if model_type in WHISPER_MODEL_TYPES:
            logger.info("Using Transformers Whisper API")
            return WhisperHFTranscriber(model_id, language, torch_dtype=kwargs.get("torch_dtype"))

        model = _load_multimodal_model(model_id, device, ov_config, True, False, **kwargs)
        return MultimodalTranscriber(model, model_id, language or "English")


class ASRGenAITranscriber:
    @staticmethod
    def create(model_id, device="CPU", ov_config=None, language="", **kwargs):
        model_type = _get_asr_model_type(model_id)
        if model_type in FUNASR_MODEL_TYPES:
            logger.info("Using OpenVINO GenAI ASRPipeline API")
            return GenAISpeechRecognitionTranscriber(model_id, device, ov_config, language or "en")
        if model_type in WHISPER_MODEL_TYPES:
            logger.info("Using OpenVINO GenAI ASRPipeline API")
            return GenAISpeechRecognitionTranscriber(model_id, device, ov_config, language)
        model = _load_multimodal_model(model_id, device, ov_config, False, True, **kwargs)
        return GenAIMultimodalTranscriber(model, language or "English")


class ASROptimumTranscriber:
    @staticmethod
    def create(model_id, device="CPU", ov_config=None, language="", **kwargs):
        model_type = _get_asr_model_type(model_id)
        if model_type in FUNASR_MODEL_TYPES:
            logger.info("Using Optimum FunASR API")
            return FunASROptimumTranscriber(model_id, model_type, device, ov_config, language or "en")
        if model_type in WHISPER_MODEL_TYPES:
            logger.info("Using Optimum Whisper API")
            return WhisperOptimumTranscriber(model_id, device, ov_config, language)

        model = _load_multimodal_model(model_id, device, ov_config, False, False, **kwargs)
        return MultimodalTranscriber(model, model_id, language or "English")
