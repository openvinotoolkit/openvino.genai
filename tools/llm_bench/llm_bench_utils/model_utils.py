# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import json
import logging as log
from pathlib import Path
from llm_bench_utils.config_class import DEFAULT_MODEL_CLASSES, USE_CASES, OV_MODEL_CLASSES_MAPPING, PT_MODEL_CLASSES_MAPPING
import librosa


def get_param_from_file(args, input_key):
    is_json_data = False
    data_list = []
    if args[input_key] is None and args['prompt_file'] is None:
        if args['use_case'] == 'text_gen':
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


def add_stateful_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--stateful',
        action='store_true',
        default=None,
        help='Replace kv-cache inputs and outputs in the model by internal variables making a stateful model. '
        'Additional operations are inserted into the model to handle cache state (Gathers, ShapeOf, etc.)',
    )

    parser.add_argument(
        '--disable-stateful',
        action="store_true",
        default=None,
        help="Disable stateful transformation for model conversion"
    )


def analyze_args(args):
    model_args = {}
    model_args['prompt'] = args.prompt
    model_args['prompt_file'] = args.prompt_file
    model_args['infer_count'] = args.infer_count
    model_args['images'] = args.images
    model_args['seed'] = args.seed
    model_args['mem_consumption'] = args.memory_consumption
    model_args['batch_size'] = args.batch_size
    model_args['fuse_decoding_strategy'] = args.fuse_decoding_strategy
    model_args['stateful'] = args.stateful
    model_args['save_prepared_model'] = args.save_prepared_model
    model_args['num_beams'] = args.num_beams
    model_args['torch_compile_backend'] = args.torch_compile_backend
    model_args['torch_compile_dynamic'] = args.torch_compile_dynamic
    model_args['torch_compile_options'] = args.torch_compile_options
    model_args['torch_compile_input_module'] = args.torch_compile_input_module
    model_args['media'] = args.media

    optimum = args.optimum

    if optimum and args.genai:
        raise RuntimeError("`--genai` and `--optimum` can not be selected in the same time")
    model_args["optimum"] = optimum
    model_args["genai"] = not optimum

    has_torch_compile_options = any([args.torch_compile_options is not None, args.torch_compile_options is not None, args.torch_compile_dynamic])
    if model_args["torch_compile_backend"] is None and has_torch_compile_options:
        log.warning("torch.compile configuration options provided, but backend is not selected, openvino backend will be used")
        model_args["torch_compile_backend"] = "openvino"
    model_args['convert_tokenizer'] = args.convert_tokenizer
    model_args['subsequent'] = args.subsequent
    model_args['output_dir'] = args.output_dir
    model_args['lora'] = args.lora
    model_args['lora_alphas'] = args.lora_alphas
    model_args["use_cb"] = args.use_cb
    model_args['devices'] = args.device
    model_args['prompt_index'] = [] if args.prompt_index is not None else None
    if model_args['prompt_index'] is not None:
        # Deduplication
        [model_args['prompt_index'].append(i) for i in args.prompt_index if i not in model_args['prompt_index']]
    model_args['end_token_stopping'] = args.end_token_stopping

    model_framework = args.framework
    model_path = Path(args.model)
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
        log.info(f"OV Config={model_args['config']}")
    elif model_framework == 'pt':
        log.info(f"PT Config={model_args['config']}")
    model_args['model_type'] = get_model_type(model_name, use_case, model_framework)
    model_args['model_name'] = model_name

    if (args.use_cb or args.draft_model) and optimum:
        raise RuntimeError("Continuous batching mode supported only via OpenVINO GenAI")
    cb_config = None
    if args.cb_config:
        cb_config = get_config(args.cb_config)
    model_args["cb_config"] = cb_config
    model_args['draft_model'] = args.draft_model
    model_args['draft_device'] = args.draft_device
    draft_cb_config = None
    if args.draft_cb_config:
        draft_cb_config = get_config(args.draft_cb_config)
    model_args["draft_cb_config"] = draft_cb_config
    model_args['num_assistant_tokens'] = args.num_assistant_tokens
    model_args['assistant_confidence_threshold'] = args.assistant_confidence_threshold
    return model_path, model_framework, model_args, model_name


def get_use_case(model_name_or_path):
    # 1. try to get use_case from model name
    path = os.path.normpath(model_name_or_path)
    model_names = path.split(os.sep)
    for model_name in reversed(model_names):
        for case, model_ids in USE_CASES.items():
            for model_id in model_ids:
                if model_name.lower().startswith(model_id):
                    log.info(f'==SUCCESS FOUND==: use_case: {case}, model_type: {model_name}')
                    return case, model_name

    # 2. try to get use_case from model config
    try:
        config_file = Path(model_name_or_path) / "config.json"
        config = json.loads(config_file.read_text())
    except Exception:
        config = None
    if (Path(model_name_or_path) / "model_index.json").exists():
        diffusers_config = json.loads((Path(model_name_or_path) / "model_index.json").read_text())
        pipe_type = diffusers_config.get("_class_name")
        if pipe_type in ["StableDiffusionPipeline", "StableDiffusionXLPipeline", "StableDiffusion3Pipeline", "FluxPipeline", "LatentConsistencyModelPipeline"]:
            return "image_gen", pipe_type.replace("Pipeline", "")

    if config is not None:
        for case, model_ids in USE_CASES.items():
            for idx, model_id in enumerate(normalize_model_ids(model_ids)):
                if config.get("model_type").lower().replace('_', '-').startswith(model_id):
                    log.info(f'==SUCCESS FOUND==: use_case: {case}, model_type: {model_id}')
                    return case, model_ids[idx]

    raise RuntimeError('==Failure FOUND==: no use_case found')


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
    precision_list = [
        'FP32', 'FP16',
        'FP16-INT8', 'INT8', 'INT8_compressed_weights', 'INT8_quantized', 'PT_compressed_weights',
        'OV_FP32-INT8', 'OV_FP16-INT8',
        'OV_FP32-INT8_ASYM', 'OV_FP32-INT8_SYM', 'OV_FP16-INT8_ASYM', 'OV_FP16-INT8_SYM',
        'PT_FP32-INT8', 'PT_FP16-INT8', 'PT_FP32-INT8_ASYM', 'PT_FP32-INT8_SYM', 'PT_FP16-INT8_ASYM', 'PT_FP16-INT8_SYM',
        'GPTQ_INT4-FP32', 'GPTQ_INT4-FP16', 'INT4',
        'OV_FP16-INT4_SYM', 'OV_FP16-INT4_ASYM', 'OV_FP32-INT4_SYM', 'OV_FP32-INT4_ASYM',
        'OV_FP32-4BIT_DEFAULT', 'OV_FP16-4BIT_DEFAULT', 'OV_FP32-4BIT_MAXIMUM', 'OV_FP16-4BIT_MAXIMUM']
    model_precision = 'unknown'
    # Search from right to left of model path
    for i in range(len(model_name_list) - 1, -1, -1):
        for precision in precision_list:
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
