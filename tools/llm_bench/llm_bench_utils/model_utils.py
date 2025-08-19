# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import json
import torch
import numpy as np
import logging as log
from pathlib import Path
from llm_bench_utils.config_class import (
    DEFAULT_MODEL_CLASSES,
    USE_CASES,
    OV_MODEL_CLASSES_MAPPING,
    PT_MODEL_CLASSES_MAPPING,
    PA_ATTENTION_BACKEND,
    SDPA_ATTENTION_BACKEND
)
import librosa

KNOWN_PRECISIONS = [
    'FP32', 'FP16',
    'FP16-INT8', 'INT8', 'INT8_compressed_weights', 'INT8_quantized', 'PT_compressed_weights',
    'OV_FP32-INT8', 'OV_FP16-INT8',
    'OV_FP32-INT8_ASYM', 'OV_FP32-INT8_SYM', 'OV_FP16-INT8_ASYM', 'OV_FP16-INT8_SYM', 'OV_FP16-INT8_ASYM_HYBRID',
    'PT_FP32-INT8', 'PT_FP16-INT8', 'PT_FP32-INT8_ASYM', 'PT_FP32-INT8_SYM', 'PT_FP16-INT8_ASYM', 'PT_FP16-INT8_SYM',
    'GPTQ_INT4-FP32', 'GPTQ_INT4-FP16', 'INT4',
    'OV_FP16-INT4_SYM', 'OV_FP16-INT4_ASYM', 'OV_FP32-INT4_SYM', 'OV_FP32-INT4_ASYM',
    'OV_FP32-4BIT_DEFAULT', 'OV_FP16-4BIT_DEFAULT', 'OV_FP32-4BIT_MAXIMUM', 'OV_FP16-4BIT_MAXIMUM']


KNOWN_FRAMEWORKS = ['pytorch', 'ov', 'dldt']


OTHER_IGNORE_MODEL_PATH_PARTS = ['compressed_weights']


def get_param_from_file(args, input_key):
    is_json_data = False
    data_list = []
    if args['prompt_file'] is None:
        if not isinstance(input_key, (list, tuple)):
            if args[input_key] is None:
                if args['use_case'] in ['text_gen', 'text_embed', 'text2speech']:
                    data_list.append('What is OpenVINO?')
                elif args['use_case'] == 'code_gen':
                    data_list.append('def print_hello_world():')
                elif args['use_case'] == 'image_gen':
                    data_list.append('sailing ship in storm by Leonardo da Vinci')
                else:
                    raise RuntimeError(f'== {input_key} and prompt file is empty ==')

            elif args[input_key] is not None and args['prompt_file'] is not None:
                raise RuntimeError(f'== {input_key} and prompt file should not exist together ==')
            else:
                if args[input_key] is not None:
                    if args[input_key] != '':
                        data_list.append(args[input_key])
                    else:
                        raise RuntimeError(f'== {input_key} path should not be empty string ==')
        else:
            if args["use_case"] != "vlm" and args["use_case"] != "image_gen":
                raise RuntimeError("Multiple sources for benchmarking supported for Visual Language Models / Image To Image Models / Inpainting Models")
            data_dict = {}
            if "media" in input_key:
                if args["media"] is None and args["images"] is None:
                    if args["use_case"] == "vlm":
                        log.warn("Input image is not provided. Only text generation part will be evaluated")
                    elif args["use_case"] != "image_gen":
                        raise RuntimeError("No input image. ImageToImage/Inpainting Models cannot start generation without one. Please, provide an image.")
                else:
                    data_dict["media"] = args["media"] if args["media"] is not None else args["images"]
            if args["prompt"] is None:
                if args["use_case"] == "vlm":
                    data_dict["prompt"] = "What is OpenVINO?" if data_dict.get("media") is None else "Describe image"
                elif args['use_case'] == 'image_gen':
                    data_dict["prompt"] = 'sailing ship in storm by Leonardo da Vinci'
            else:
                data_dict["prompt"] = args["prompt"]
            if "mask_image" in input_key:
                if args.get("mask_image"):
                    data_dict["mask_image"] = args["mask_image"]
                else:
                    raise RuntimeError("Mask image is not provided. Inpainting Models cannot start of generation wihtout it. Please, provide a mask image.")
            data_list.append(data_dict)
    else:
        input_prompt_list = args['prompt_file']
        is_json_data = True
        for input_prompt in input_prompt_list:
            if input_prompt.endswith('.jsonl'):
                if os.path.exists(input_prompt):
                    log.info(f'Read prompts from {input_prompt}')
                    with open(input_prompt, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line)
                            data_list.append(data)
                else:
                    raise RuntimeError(f'== The prompt file:{input_prompt} does not exist ==')
            else:
                raise RuntimeError(f'== The prompt file:{input_prompt} should be ended with .jsonl ==')
    return data_list, is_json_data


def read_wav(filepath, sampling_rate):
    raw_speech = librosa.load(filepath, sr=sampling_rate)
    return raw_speech[0]


def set_default_param_for_ov_config(ov_config):
    # With this PR https://github.com/huggingface/optimum-intel/pull/362, we are able to disable model cache
    if 'CACHE_DIR' not in ov_config:
        ov_config['CACHE_DIR'] = ''


def analyze_args(args):
    model_args = {}
    model_args['prompt'] = args.prompt
    model_args['prompt_file'] = args.prompt_file
    model_args['infer_count'] = args.infer_count
    model_args["num_steps"] = args.num_steps
    model_args["height"] = args.height
    model_args["width"] = args.width
    model_args['images'] = args.images
    model_args['seed'] = args.seed
    model_args['mem_consumption'] = args.memory_consumption
    model_args['batch_size'] = args.batch_size
    model_args['num_beams'] = args.num_beams
    model_args['torch_compile_backend'] = args.torch_compile_backend
    model_args['torch_compile_dynamic'] = args.torch_compile_dynamic
    model_args['torch_compile_options'] = args.torch_compile_options
    model_args['torch_compile_input_module'] = args.torch_compile_input_module
    model_args['media'] = args.media
    model_args["disable_prompt_permutation"] = args.disable_prompt_permutation
    model_args["static_reshape"] = args.static_reshape
    model_args['mask_image'] = args.mask_image
    model_args['task'] = args.task
    model_args['strength'] = args.strength
    model_args['emb_pooling_type'] = args.embedding_pooling
    model_args['emb_normalize'] = args.embedding_normalize
    model_args["emb_max_length"] = args.embedding_max_length
    model_args["apply_chat_template"] = args.apply_chat_template

    # CDPruner config
    model_args['enable_pruning'] = args.enable_pruning
    log.info(f"CDPruner config: Enable pruning - {model_args['enable_pruning']}")
    if model_args['enable_pruning']:
        if args.visual_tokens_retain_percentage is not None:
            model_args['visual_tokens_retain_percentage'] = args.visual_tokens_retain_percentage
        if args.relevance_weight is not None:
            model_args['relevance_weight'] = args.relevance_weight
        if args.pruning_debug_mode:
            model_args['pruning_debug_mode'] = args.pruning_debug_mode
        log.info(f"CDPruner config: Percentage of visual tokens to keep - {model_args['visual_tokens_retain_percentage']}%")
        log.info(f"CDPruner config: Pruning debug mode - {model_args['pruning_debug_mode']}")

    optimum = args.optimum

    if optimum and args.genai:
        raise RuntimeError("`--genai` and `--optimum` can not be selected in the same time")
    if args.from_onnx and not optimum:
        log.warning("ONNX model initialization supported only using Optimum. Benchmarking will be switched to this backend")
        optimum = True
    model_args["optimum"] = optimum
    model_args["genai"] = not optimum
    model_args["from_onnx"] = args.from_onnx

    has_torch_compile_options = any([args.torch_compile_options is not None, args.torch_compile_options is not None, args.torch_compile_dynamic])
    if model_args["torch_compile_backend"] is None and has_torch_compile_options:
        log.warning("torch.compile configuration options provided, but backend is not selected, openvino backend will be used")
        model_args["torch_compile_backend"] = "openvino"
    model_args['convert_tokenizer'] = args.convert_tokenizer
    model_args['subsequent'] = args.subsequent
    model_args['output_dir'] = args.output_dir
    model_args['lora'] = args.lora
    model_args['lora_alphas'] = args.lora_alphas
    model_args['lora_mode'] = args.lora_mode
    model_args['empty_lora'] = args.empty_lora
    model_args['devices'] = args.device
    model_args['prompt_index'] = [] if args.prompt_index is not None else None
    if model_args['prompt_index'] is not None:
        # Deduplication
        [model_args['prompt_index'].append(i) for i in args.prompt_index if i not in model_args['prompt_index']]
    model_args['end_token_stopping'] = args.end_token_stopping

    model_framework = args.framework
    model_path = Path(args.model)
    if model_args["torch_compile_backend"]:
        log.info("Setting Framework to PyTorch Since torch_compile_backend is provided.")
        model_framework = 'pt'
    if not model_path.exists():
        raise RuntimeError(f'==Failure FOUND==: Incorrect model path:{model_path}')
    if model_framework in ('ov', 'pt'):
        use_case, model_name = get_use_case(args.model)
    model_args['use_case'] = use_case
    if use_case == 'code_gen' and not model_args['prompt'] and not model_args['prompt_file']:
        model_args['prompt'] = 'def print_hello_world():'
    model_args['config'] = {}
    if args.load_config is not None:
        config = get_config(args.load_config)
        if type(config) is dict and len(config) > 0:
            model_args['config'] = config
    if model_framework == 'ov':
        set_default_param_for_ov_config(model_args['config'])
        if 'ATTENTION_BACKEND' not in model_args['config'] and not optimum and args.device != "NPU":
            if use_case in ['text_gen']:
                model_args['config']['ATTENTION_BACKEND'] = PA_ATTENTION_BACKEND
            elif use_case in ['vlm']:
                model_args['config']['ATTENTION_BACKEND'] = SDPA_ATTENTION_BACKEND
        log.info(f"OV Config={model_args['config']}")
    elif model_framework == 'pt':
        log.info(f"PT Config={model_args['config']}")
    model_args['model_type'] = get_model_type(model_name, use_case, model_framework)
    model_args['model_name'] = model_name

    cb_config = None
    if args.cb_config:
        cb_config = get_config(args.cb_config)
    model_args["cb_config"] = cb_config
    if args.draft_model and (args.device == "NPU" or model_args['config']['ATTENTION_BACKEND'] != PA_ATTENTION_BACKEND):
        log.warning("Speculative Decoding is supported only with Page Attention Backend and not supported for NPU device")
        args.draft_model = None
    model_args['draft_model'] = args.draft_model
    model_args['draft_device'] = args.draft_device
    draft_cb_config = None
    if args.draft_cb_config:
        draft_cb_config = get_config(args.draft_cb_config)
    model_args["draft_cb_config"] = draft_cb_config
    model_args['num_assistant_tokens'] = args.num_assistant_tokens
    model_args['assistant_confidence_threshold'] = args.assistant_confidence_threshold
    model_args['max_ngram_size'] = args.max_ngram_size

    model_args['speaker_embeddings'] = None
    if args.speaker_embeddings:
        model_args['speaker_embeddings'] = get_speaker_embeddings(args.speaker_embeddings)
    model_args['vocoder_path'] = args.vocoder_path
    if model_args['vocoder_path'] and not Path(model_args['vocoder_path']).exists():
        raise RuntimeError(f'==Failure FOUND==: Incorrect vocoder path:{model_args["vocoder_path"]}')

    return model_path, model_framework, model_args, model_name


def get_use_case(model_name_or_path):
    config_file = Path(model_name_or_path) / "config.json"
    config = None
    if config_file.exists():
        config = json.loads(config_file.read_text())
    if (Path(model_name_or_path) / "model_index.json").exists():
        diffusers_config = json.loads((Path(model_name_or_path) / "model_index.json").read_text())
        pipe_type = diffusers_config.get("_class_name")
        if pipe_type in ["StableDiffusionPipeline", "StableDiffusionXLPipeline", "StableDiffusion3Pipeline", "StableDiffusionInpaintPipeline",
                         "StableDiffusionXLInpaintPipeline", "FluxPipeline", "LatentConsistencyModelPipeline"]:
            return "image_gen", pipe_type.replace("Pipeline", "")

    if config is not None:
        case, model_name = resolve_complex_model_types(config)
        if case is not None:
            log.info(f'==SUCCESS FOUND==: use_case: {case}, model_type: {model_name}')
            return case, model_name
        for case, model_ids in USE_CASES.items():
            for idx, model_id in enumerate(normalize_model_ids(model_ids)):
                if config.get("model_type").lower().replace('_', '-').startswith(model_id):
                    log.info(f'==SUCCESS FOUND==: use_case: {case}, model_type: {model_id}')
                    return case, model_ids[idx]

    case, model_name = get_model_name(model_name_or_path)
    if case is None:
        raise RuntimeError('==Failure FOUND==: no use_case found')
    else:
        log.info(f'==SUCCESS FOUND==: use_case: {case}, model_Name: {model_name}')
    return case, model_name


def resolve_complex_model_types(config):
    model_type = config.get("model_type").lower().replace('_', '-')
    if model_type == "gemma3":
        return "vlm", model_type
    if model_type == "gemma3-text":
        return "text_gen", model_type
    if model_type in ["phi4mm", "phi4-multimodal"]:
        return "vlm", model_type
    if model_type == "llama4":
        return "vlm", model_type
    return None, None


def get_model_name(model_name_or_path):
    # try to get use_case from model name
    path = os.path.abspath(model_name_or_path)
    model_names = path.split(os.sep)
    for model_name in reversed(model_names):
        for case, model_ids in USE_CASES.items():
            for model_id in model_ids:
                if model_name.lower().startswith(model_id):
                    return case, model_name
    return None, None


def get_model_name_with_path_part(model_name_or_path):
    IGNORE_MODEL_PATH_PARTS = [x.lower() for x in (KNOWN_FRAMEWORKS + KNOWN_PRECISIONS + OTHER_IGNORE_MODEL_PATH_PARTS)]
    model_path = Path(model_name_or_path)
    model_name = None
    for path_part in reversed(model_path.parts):
        if not path_part.lower() in IGNORE_MODEL_PATH_PARTS:
            model_name = path_part
            break
    return model_name


def get_config(config):
    if Path(config).is_file():
        with open(config, 'r') as f:
            try:
                ov_config = json.load(f)
            except Exception:
                raise RuntimeError(f'==Parse file:{config} failure, json format is incorrect ==')
    else:
        try:
            ov_config = json.loads(config)
        except Exception:
            raise RuntimeError(f'==Parse config:{config} failure, json format is incorrect ==')
    return ov_config


def get_model_type(model_name, use_case, model_framework):
    default_model_type = DEFAULT_MODEL_CLASSES.get(use_case)
    if model_framework == 'ov':
        for cls in OV_MODEL_CLASSES_MAPPING:
            if cls in model_name.lower():
                return cls
    elif model_framework == 'pt':
        for cls in PT_MODEL_CLASSES_MAPPING:
            if cls in model_name.lower():
                return cls
    return default_model_type


def normalize_model_ids(model_ids_list):
    return [m_id[:-1] if m_id.endswith('_') else m_id for m_id in model_ids_list]


def get_ir_conversion_frontend(cur_model_name, model_name_list):
    ir_conversion_frontend = ''
    idx = 0
    for model_name in model_name_list:
        # idx+1 < len(model_name_list) to avoid out of bounds index of model_name_list
        if model_name == cur_model_name and idx + 1 < len(model_name_list):
            ir_conversion_frontend = model_name_list[idx + 1]
            break
        idx = idx + 1
    return ir_conversion_frontend


def get_model_precision(model_name_list):
    model_precision = 'unknown'
    # Search from right to left of model path
    for i in range(len(model_name_list) - 1, -1, -1):
        for precision in KNOWN_PRECISIONS:
            if model_name_list[i] == precision:
                model_precision = precision
                break
        if model_precision != 'unknown':
            break
    return '' if model_precision == 'unknown' else model_precision


def init_timestamp(num_iters, prompt_list, prompt_idx_list):
    iter_timestamp = {}
    for num in range(num_iters + 1):
        iter_timestamp[num] = {}
        for idx, input_text in enumerate(prompt_list):
            p_idx = prompt_idx_list[idx]
            iter_timestamp[num][p_idx] = {}
    return iter_timestamp


def resolve_media_file_path(file_path, prompt_file_path):
    if not file_path:
        return file_path
    if not (file_path.startswith("http://") or file_path.startswith("https://")):
        return os.path.join(os.path.dirname(prompt_file_path), file_path.replace("./", ""))
    return file_path


def get_version_in_format_to_pars(version):
    processed_version = version
    if "-" in processed_version:
        ov_major_version, dev_info = version.split("-", 1)
        commit_id = dev_info.split("-")[0]
        processed_version = f"{ov_major_version}-{commit_id}"
    return processed_version


def get_speaker_embeddings(speaker_embeddings_file, expected_shape=(1, 512)):
    speaker_embeddings = None
    if Path(speaker_embeddings_file).is_file():
        if ('.pt' in speaker_embeddings_file):
            try:
                speaker_embeddings = torch.load(speaker_embeddings_file)
            except Exception:
                raise RuntimeError(f'==Parse file with torch: {speaker_embeddings_file} failure, format is incorrect ==')
        else:
            try:
                speaker_embeddings = np.fromfile(speaker_embeddings_file, dtype=np.float32)
                speaker_embeddings = torch.from_numpy(speaker_embeddings)
            except Exception:
                raise RuntimeError(f'==Parse config: {speaker_embeddings_file} failure, format is incorrect ==')
        if speaker_embeddings.numel() != np.prod(expected_shape):
            raise RuntimeError(f"==Expected {np.prod(expected_shape)} elements, but got {speaker_embeddings.numel()} in {speaker_embeddings_file}==")
        speaker_embeddings = speaker_embeddings.reshape(expected_shape)
    else:
        raise RuntimeError(f'==Failure FOUND==: Incorrect speaker embeddings file path:{speaker_embeddings_file}')

    return speaker_embeddings
