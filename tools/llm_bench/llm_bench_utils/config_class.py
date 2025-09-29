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


def normalize_model_ids(model_ids_list):
    return [m_id[:-1] if m_id.endswith('_') else m_id for m_id in model_ids_list]


def get_use_case_by_model_id(model_type, task=None):
    possible_use_cases = sum(list(USE_CASES.values()), [])
    if task:
        if task in list(UseCaseImageGen.TASK.keys()):
            possible_use_cases = USE_CASES["image_gen"]
        else:
            possible_use_cases = USE_CASES[task]
    for use_case in possible_use_cases:
        for m_type in normalize_model_ids(use_case.model_types):
            # TODO go to equality and raise error if use_cases is already found, as it will mean that
            # model with that task can be applicable to execute with different pipelines and user doesn't specify one
            if model_type.startswith(m_type):
                return use_case, m_type

    return None, None


class UseCase:
    task = None
    model_types = []
    ov_cls = None
    pt_cls = AutoModel
    tokenizer_cls = AutoTokenizer

    def __init__(self, model_types, ov_cls=None, pt_cls=None, tokenizer_cls=None):
        self.model_types = model_types
        self.ov_cls = ov_cls if ov_cls else self.ov_cls
        self.pt_cls = pt_cls if pt_cls else self.pt_cls
        self.def_tokenizer = tokenizer_cls if tokenizer_cls else self.tokenizer_cls

    def eq_use_case(self, user_model_id, user_task=None):
        return ((user_task is None or user_task == self.task) and user_model_id in self.model_types)


class UseCaseImageGen(UseCase):
    task = "image_gen"
    ov_cls = OVDiffusionPipeline
    pt_cls = DiffusionPipeline

    TASK = {
        "text2img": {"name": 'text-to-image', "ov_cls": OVDiffusionPipeline},
        "img2img": {"name": 'image-to-image', "ov_cls": OVPipelineForImage2Image},
        "inpainting": {"name": 'inpainting', "ov_cls": OVPipelineForInpainting}
    }


class UseCaseVLM(UseCase):
    task = "visual_text_gen"
    ov_cls = OVModelForVisualCausalLM
    pt_cls = None


class UseCaseSpeech2Text(UseCase):
    task = "speech_to_text"
    ov_cls = OVModelForSpeechSeq2Seq
    pt_cls = None


class UseCaseTextGen(UseCase):
    task = "text_gen"
    ov_cls = OVModelForCausalLM
    pt_cls = AutoModelForCausalLM


class UseCaseCodeGen(UseCase):
    task = 'code_gen'
    ov_cls = OVModelForCausalLM
    pt_cls = AutoModelForCausalLM


class UseCaseImageCls(UseCase):
    task = 'image_cls'
    ov_cls = OVModelForCausalLM
    pt_cls = AutoModelForCausalLM


class UseCaseLDMSuperResolution(UseCase):
    task = 'ldm_super_resolution'
    ov_cls = OVLDMSuperResolutionPipeline
    pt_cls = LDMSuperResolutionPipeline


class UseCaseTextEmbeddings(UseCase):
    task = 'text_embed'
    ov_cls = OVModelForFeatureExtraction
    pt_cls = AutoModel


class UseCaseTextReranker(UseCase):
    task = 'text_rerank'
    ov_cls = OVModelForSequenceClassification
    pt_cls = AutoModelForSequenceClassification


class UseCaseTextToSpeech(UseCase):
    task = 'text_to_speech'
    ov_cls = OVModelForTextToSpeechSeq2Seq
    pt_cls = SpeechT5ForTextToSpeech
    tokenizer_cls = SpeechT5Processor
    vocoder_cls = SpeechT5HifiGan


USE_CASES = {
    'image_gen': [UseCaseImageGen(['stable-diffusion-', 'ssd-', 'tiny-sd', 'small-sd', 'lcm-', 'sdxl', 'dreamlike', "flux"])],
    "visual_text_gen": [UseCaseVLM(["llava", "llava-next", "qwen2-vl", "llava-qwen2", "internvl-chat", "minicpmv", "phi3-v",
                                    "minicpm-v", "maira2", "qwen2-5-vl"])],
    'speech_to_text': [UseCaseSpeech2Text(['whisper'])],
    'image_cls': [UseCaseImageCls(['vit'])],
    'code_gen': [UseCaseCodeGen(["codegen", "codegen2", "stable-code"]),
                 UseCaseCodeGen(['replit'], OVMPTModel),
                 UseCaseCodeGen(['codet5'], ov_cls=OVModelForSeq2SeqLM)],
    'text_gen': [UseCaseTextGen(['arcee', "decoder", "falcon", "glm", "aquila", "gpt2", "open-llama", "openchat", "neural-chat", "llama",
                                 "tiny-llama", "tinyllama", "opt", "opt-", "pythia", "pythia-", "stablelm", "stablelm-", "stable-zephyr-", "rocket-",
                                 "vicuna", "dolly", "bloom", "red-pajama", "xgen", "longchat", "jais", "orca-mini", "baichuan", "qwen", "zephyr",
                                 "mistral", "mixtral", "phi2-", "minicpm", "gemma", "deci", "phi3", "deci", "internlm", "olmo", "starcoder", "instruct-gpt",
                                 "granite", "granitemoe", "gptj"]),
                 UseCaseTextGen(['t5'], ov_cls=OVModelForSeq2SeqLM, pt_cls=T5ForConditionalGeneration),
                 UseCaseTextGen(["gpt", "gpt-"], ov_cls=OVModelForSeq2SeqLM),
                 UseCaseTextGen(['mpt'], OVMPTModel),
                 UseCaseTextGen(['blenderbot'], ov_cls=OVModelForSeq2SeqLM, pt_cls=BlenderbotForConditionalGeneration),
                 UseCaseTextGen(['chatglm'], ov_cls=OVChatGLMModel, pt_cls=AutoModel),
                 UseCaseTextGen(['yi-'], ov_cls=OVModelForSeq2SeqLM),
                 UseCaseTextGen(["phi", "phi-"], ov_cls=OVModelForSeq2SeqLM)],
    'ldm_super_resolution': [UseCaseLDMSuperResolution(['ldm-super-resolution'])],
    'text_embed': [UseCaseTextEmbeddings(["qwen3", "bge", "bert", "albert", "roberta", "xlm-roberta"])],
    'text_rerank': [UseCaseTextReranker(["qwen3", "bge", "bert", "albert", "roberta", "xlm-roberta"])],
    'text_to_speech': [UseCaseTextToSpeech(['speecht5'])],
}

PA_ATTENTION_BACKEND = "PA"
SDPA_ATTENTION_BACKEND = "SDPA"
