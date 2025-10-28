# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from transformers import AutoTokenizer
from transformers import (
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    BlenderbotForConditionalGeneration,
    AutoModel,
    SpeechT5ForTextToSpeech,
    SpeechT5Processor,
    SpeechT5HifiGan,
    AutoModelForSequenceClassification
)
from diffusers.pipelines import DiffusionPipeline, LDMSuperResolutionPipeline
from optimum.intel.openvino import (
    OVModelForCausalLM,
    OVModelForSeq2SeqLM,
    OVDiffusionPipeline,
    OVModelForSpeechSeq2Seq,
    OVModelForVisualCausalLM,
    OVPipelineForInpainting,
    OVPipelineForImage2Image,
    OVModelForFeatureExtraction,
    OVModelForTextToSpeechSeq2Seq,
    OVModelForSequenceClassification
)
from llm_bench_utils.ov_model_classes import OVMPTModel, OVLDMSuperResolutionPipeline, OVChatGLMModel
from dataclasses import dataclass, field


@dataclass
class UseCase:
    task = ''
    model_types: list[str] = field(default_factory=list)
    ov_cls: type | None = None
    pt_cls: type | None = AutoModel
    tokenizer_cls: type = AutoTokenizer


@dataclass
class UseCaseImageGen(UseCase):
    task = "image_gen"
    ov_cls: type | None = OVDiffusionPipeline
    pt_cls: type | None = DiffusionPipeline

    TASK = {
        "text2img": {"name": 'text-to-image', "ov_cls": OVDiffusionPipeline},
        "img2img": {"name": 'image-to-image', "ov_cls": OVPipelineForImage2Image},
        "inpainting": {"name": 'inpainting', "ov_cls": OVPipelineForInpainting}
    }


@dataclass
class UseCaseVLM(UseCase):
    task = "visual_text_gen"
    ov_cls: type | None = OVModelForVisualCausalLM
    pt_cls: type | None = None


@dataclass
class UseCaseSpeech2Text(UseCase):
    task = "speech_to_text"
    ov_cls: type | None = OVModelForSpeechSeq2Seq
    pt_cls: type | None = None


@dataclass
class UseCaseTextGen(UseCase):
    task = "text_gen"
    ov_cls: type | None = OVModelForCausalLM
    pt_cls: type | None = AutoModelForCausalLM


@dataclass
class UseCaseCodeGen(UseCase):
    task = 'code_gen'
    ov_cls: type | None = OVModelForCausalLM
    pt_cls: type | None = AutoModelForCausalLM


@dataclass
class UseCaseImageCls(UseCase):
    task = 'image_cls'
    ov_cls: type | None = OVModelForCausalLM
    pt_cls: type | None = AutoModelForCausalLM


@dataclass
class UseCaseLDMSuperResolution(UseCase):
    task = 'ldm_super_resolution'
    ov_cls: type | None = OVLDMSuperResolutionPipeline
    pt_cls: type | None = LDMSuperResolutionPipeline


@dataclass
class UseCaseTextEmbeddings(UseCase):
    task = 'text_embed'
    ov_cls: type | None = OVModelForFeatureExtraction
    pt_cls: type | None = AutoModel


@dataclass
class UseCaseTextReranker(UseCase):
    task = 'text_rerank'
    ov_cls: type | None = OVModelForSequenceClassification
    pt_cls: type | None = AutoModelForSequenceClassification

    def adjust_model_class_by_config(self, config):
        if self.is_qwen_causallm_arch(config):
            self.ov_cls = OVModelForCausalLM
            self.pt_cls = AutoModelForCausalLM

    @staticmethod
    def is_qwen_causallm_arch(config):
        return config.model_type == "qwen3" and "Qwen3ForCausalLM" in config.architectures


@dataclass
class UseCaseTextToSpeech(UseCase):
    task = 'text_to_speech'
    ov_cls: type | None = OVModelForTextToSpeechSeq2Seq
    pt_cls: type | None = SpeechT5ForTextToSpeech
    tokenizer_cls: type = SpeechT5Processor
    vocoder_cls: type = SpeechT5HifiGan


USE_CASES = {
    'image_gen': [UseCaseImageGen(['stable-diffusion-', 'ssd-', 'tiny-sd', 'small-sd', 'lcm-', 'sdxl', 'dreamlike', "flux"])],
    "visual_text_gen": [UseCaseVLM(["llava", "llava-next", "qwen2-vl", "llava-qwen2", "internvl-chat", "minicpmv", "phi3-v",
                                    "minicpm-v", "minicpmo", "maira2", "qwen2-5-vl"])],
    'speech_to_text': [UseCaseSpeech2Text(['whisper'])],
    'image_cls': [UseCaseImageCls(['vit'])],
    'code_gen': [UseCaseCodeGen(["codegen", "codegen2", "stable-code"]),
                 UseCaseCodeGen(['replit'], ov_cls=OVMPTModel),
                 UseCaseCodeGen(['codet5'], ov_cls=OVModelForSeq2SeqLM)],
    'text_gen': [UseCaseTextGen(['arcee', "decoder", "falcon", "glm", "aquila", "gpt", "gpt-", "gpt2", "open-llama", "openchat", "neural-chat", "llama",
                                 "tiny-llama", "tinyllama", "opt", "opt-", "pythia", "pythia-", "stablelm", "stablelm-", "stable-zephyr-", "rocket-",
                                 "vicuna", "dolly", "bloom", "red-pajama", "xgen", "longchat", "jais", "orca-mini", "baichuan", "qwen", "zephyr",
                                 "mistral", "mixtral", "phi", "phi2-", "minicpm", "gemma", "deci", "phi3", "internlm", "olmo", "starcoder", "instruct-gpt",
                                 "granite", "granitemoe", "gptj", "yi-"]),
                 UseCaseTextGen(['t5'], ov_cls=OVModelForSeq2SeqLM, pt_cls=T5ForConditionalGeneration),
                 UseCaseTextGen(['mpt'], OVMPTModel),
                 UseCaseTextGen(['blenderbot'], ov_cls=OVModelForSeq2SeqLM, pt_cls=BlenderbotForConditionalGeneration),
                 UseCaseTextGen(['chatglm'], ov_cls=OVChatGLMModel, pt_cls=AutoModel)],
    'ldm_super_resolution': [UseCaseLDMSuperResolution(['ldm-super-resolution'])],
    'text_embed': [UseCaseTextEmbeddings(["qwen3", "bge", "bert", "albert", "roberta", "xlm-roberta"])],
    'text_rerank': [UseCaseTextReranker(["qwen3", "bge", "bert", "albert", "roberta", "xlm-roberta"])],
    'text_to_speech': [UseCaseTextToSpeech(['speecht5'])],
}

PA_ATTENTION_BACKEND = "PA"
SDPA_ATTENTION_BACKEND = "SDPA"
