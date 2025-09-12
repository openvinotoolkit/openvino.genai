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
    SpeechT5HifiGan
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
    OVModelForTextToSpeechSeq2Seq
)
from llm_bench_utils.ov_model_classes import OVMPTModel, OVLDMSuperResolutionPipeline, OVChatGLMModel

TOKENIZE_CLASSES_MAPPING = {
    'decoder': AutoTokenizer,
    'mpt': AutoTokenizer,
    't5': AutoTokenizer,
    'blenderbot': AutoTokenizer,
    'falcon': AutoTokenizer,
    'speecht5': SpeechT5Processor
}

TEXT_TO_SPEECH_VOCODER_CLS = SpeechT5HifiGan

IMAGE_GEN_CLS = OVDiffusionPipeline

INPAINTING_IMAGE_GEN_CLS = OVPipelineForInpainting

IMAGE_TO_IMAGE_GEN_CLS = OVPipelineForImage2Image

OV_MODEL_CLASSES_MAPPING = {
    'decoder': OVModelForCausalLM,
    't5': OVModelForSeq2SeqLM,
    'blenderbot': OVModelForSeq2SeqLM,
    'falcon': OVModelForCausalLM,
    'mpt': OVMPTModel,
    'replit': OVMPTModel,
    'codet5': OVModelForSeq2SeqLM,
    'codegen2': OVModelForCausalLM,
    'ldm_super_resolution': OVLDMSuperResolutionPipeline,
    'chatglm2': OVModelForCausalLM,
    'chatglm3': OVModelForCausalLM,
    'chatglm': OVChatGLMModel,
    'whisper': OVModelForSpeechSeq2Seq,
    "vlm": OVModelForVisualCausalLM,
    "bert": OVModelForFeatureExtraction,
    'speecht5': OVModelForTextToSpeechSeq2Seq
}

PT_MODEL_CLASSES_MAPPING = {
    'decoder': AutoModelForCausalLM,
    't5': T5ForConditionalGeneration,
    'blenderbot': BlenderbotForConditionalGeneration,
    'mpt': AutoModelForCausalLM,
    'falcon': AutoModelForCausalLM,
    'stable_diffusion': DiffusionPipeline,
    'ldm_super_resolution': LDMSuperResolutionPipeline,
    'chatglm': AutoModel,
    "bert": AutoModel,
    'speecht5': SpeechT5ForTextToSpeech
}

USE_CASES = {
    'image_gen': ['stable-diffusion-', 'ssd-', 'tiny-sd', 'small-sd', 'lcm-', 'sdxl', 'dreamlike', "flux"],
    "vlm": ["llava", "llava-next", "qwen2-vl", "llava-qwen2", "internvl-chat", "minicpmv", "phi3-v", "minicpm-v", "maira2", "qwen2-5-vl"],
    'speech2text': ['whisper'],
    'image_cls': ['vit'],
    'code_gen': ['replit', 'codegen2', 'codegen', 'codet5', "stable-code"],
    'text_gen': [
        'arcee',
        'decoder',
        't5',
        'falcon',
        "glm",
        "gpt",
        'gpt-',
        'gpt2',
        'aquila',
        'mpt',
        'open-llama',
        'openchat',
        'neural-chat',
        'llama',
        'tiny-llama',
        'tinyllama',
        "opt",
        'opt-',
        'pythia-',
        'stablelm-',
        'stablelm',
        'stable-zephyr-',
        'rocket-',
        'blenderbot',
        'vicuna',
        'dolly',
        'bloom',
        'red-pajama',
        'chatglm',
        'xgen',
        'longchat',
        'jais',
        'orca-mini',
        'baichuan',
        'qwen',
        'zephyr',
        'mistral',
        'mixtral',
        'yi-',
        "phi",
        'phi-',
        'phi2-',
        'minicpm',
        'gemma',
        "deci",
        "internlm",
        "olmo",
        "phi3",
        "starcoder",
        "instruct-gpt",
        "granite",
        "granitemoe",
        "gptj"
    ],
    'ldm_super_resolution': ['ldm-super-resolution'],
    'text_embed': ["bge", "bert", "albert", "roberta", "xlm-roberta"],
    'text2speech': ['speecht5'],
}

DEFAULT_MODEL_CLASSES = {
    'text_gen': 'decoder',
    'image_gen': 'stable_diffusion',
    'image_cls': 'vit',
    'speech2text': 'whisper',
    'code_gen': 'decoder',
    'ldm_super_resolution': 'ldm_super_resolution',
    "vlm": "vlm",
    'text_embed': 'bert',
    'text2speech': 'speecht5',
}

TASK = {
    "img2img": "image-to-image",
    "text2img": "text-to-image",
    "inpainting": "inpainting"
}

PA_ATTENTION_BACKEND = "PA"
SDPA_ATTENTION_BACKEND = "SDPA"
