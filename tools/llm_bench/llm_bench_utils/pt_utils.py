# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import time
import torch
import json
import logging as log
from pathlib import Path
from transformers import AutoConfig

from llm_bench_utils.memory_monitor import MemMonitorWrapper
import llm_bench_utils.hook_common as hook_common


def set_bf16(model, device, **kwargs):
    try:
        if len(kwargs['config']) > 0 and kwargs['config'].get('PREC_BF16') and kwargs['config']['PREC_BF16'] is True:
            model = model.to(device.lower(), dtype=torch.bfloat16)
            log.info('Set inference precision to bf16')
    except Exception:
        log.error('Catch exception for setting inference precision to bf16.')
        raise RuntimeError('Set prec_bf16 fail.')
    return model


def torch_compile_child_module(model, child_modules, backend='openvino', dynamic=None, options=None):
    if len(child_modules) == 1:
        setattr(model, child_modules[0], torch.compile(getattr(model, child_modules[0]), backend=backend, dynamic=dynamic, fullgraph=True, options=options))
        return model
    setattr(model, child_modules[0], torch_compile_child_module(getattr(model, child_modules[0]), child_modules[1:], backend, dynamic, options))
    return model


def run_torch_compile(model, backend='openvino', dynamic=None, options=None, child_modules=None, memory_data_collector=None):
    if memory_data_collector:
        memory_data_collector.start()
    if backend == 'pytorch':
        log.info(f'Running torch.compile() with {backend} backend')
        start = time.perf_counter()
        compiled_model = torch.compile(model)
        end = time.perf_counter()
        compile_time = end - start
        log.info(f'Compiling model via torch.compile() took: {compile_time}')
    else:
        log.info(f'Running torch.compile() with {backend} backend')
        start = time.perf_counter()
        if child_modules and len(child_modules) > 0:
            compiled_model = torch_compile_child_module(model, child_modules, backend, dynamic, options)
        else:
            compiled_model = torch.compile(model, backend=backend, dynamic=dynamic, options=options)
        end = time.perf_counter()
        compile_time = end - start
        log.info(f'Compiling model via torch.compile() took: {compile_time}')
    if memory_data_collector:
        memory_data_collector.stop_and_collect_data('compilation_phase')
        memory_data_collector.log_data(compilation_phase=True)
    return compiled_model


def create_text_gen_model(model_path, device, memory_data_collector, **kwargs):
    model_path = Path(model_path)
    is_gguf_model = model_path.suffix == '.gguf'
    if not model_path.exists():
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not exist')
    if not is_gguf_model and not (model_path.is_dir() and len(os.listdir(model_path)) != 0):
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not directory or directory is empty')
    if not device:
        raise RuntimeError('==Failure ==: no device to load')

    log.info(f'Load text model from model path:{model_path}')
    model_class = kwargs['use_case'].pt_cls
    token_class = kwargs['use_case'].tokenizer_cls
    if kwargs.get("mem_consumption"):
        memory_data_collector.start()
    start = time.perf_counter()
    load_model_kwargs = {'trust_remote_code': False}
    if is_gguf_model:
        load_model_kwargs |= {'gguf_file': str(model_path)}
        model_path = model_path.parent
    try:
        model = model_class.from_pretrained(model_path, **load_model_kwargs)
    except Exception:
        start = time.perf_counter()
        load_model_kwargs['trust_remote_code'] = True
        model = model_class.from_pretrained(model_path, **load_model_kwargs)
    tokenizer = token_class.from_pretrained(model_path, **load_model_kwargs)
    end = time.perf_counter()
    from_pretrain_time = end - start
    if kwargs.get("mem_consumption"):
        memory_data_collector.stop_and_collect_data('from_pretrained_phase')
        memory_data_collector.log_data(compilation_phase=True)

    log.info(f'model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s')

    gptjfclm = 'transformers.models.gptj.modeling_gptj.GPTJForCausalLM'
    lfclm = 'transformers.models.llama.modeling_llama.LlamaForCausalLM'
    bfclm = 'transformers.models.bloom.modeling_bloom.BloomForCausalLM'
    gpt2lmhm = 'transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel'
    gptneoxclm = 'transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM'
    chatglmfcg = 'transformers_modules.pytorch_original.modeling_chatglm.ChatGLMForConditionalGeneration'
    real_base_model_name = str(type(model)).lower()
    log.info(f'Real base model={real_base_model_name}')
    # bfclm will trigger generate crash.

    # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
    if device.upper() == 'GPU':
        device = torch.device('cuda') if torch.cuda.is_available() else log.info('CUDA device is unavailable')
    else:
        device = torch.device(device.lower())
    log.info(f'Torch device was set to: {device}')

    if any(x in real_base_model_name for x in [gptjfclm, lfclm, bfclm, gpt2lmhm, gptneoxclm, chatglmfcg]):
        model = set_bf16(model, device, **kwargs)
    else:
        if len(kwargs['config']) > 0 and kwargs['config'].get('PREC_BF16') and kwargs['config']['PREC_BF16'] is True:
            log.info('Param [bf16/prec_bf16] will not work.')
        model.to(device)

    bench_hook = hook_common.get_bench_hook(kwargs['num_beams'], model)

    if kwargs['torch_compile_backend']:
        backend = kwargs['torch_compile_backend']
        dynamic = None
        options = None
        child_modules = None
        if kwargs['torch_compile_dynamic']:
            dynamic = kwargs['torch_compile_dynamic']
        if kwargs['torch_compile_options']:
            options = json.loads(kwargs['torch_compile_options'])
        if kwargs['torch_compile_input_module']:
            child_modules = kwargs['torch_compile_input_module'].split(".")
        compiled_model = run_torch_compile(model, backend, dynamic, options, child_modules, memory_data_collector if kwargs.get("mem_consumption") else None)
        model = compiled_model
    return model, tokenizer, from_pretrain_time, bench_hook, False


def create_image_gen_model(model_path, device, memory_data_collector, **kwargs):
    model_path = Path(model_path)
    from_pretrain_time = 0
    if model_path.exists():
        if model_path.is_dir() and len(os.listdir(model_path)) != 0:
            log.info(f'Load image model from model path:{model_path}')
            model_class = kwargs['use_case'].pt_cls
            if kwargs.get("mem_consumption"):
                memory_data_collector.start()
            start = time.perf_counter()
            pipe = model_class.from_pretrained(model_path)
            pipe = set_bf16(pipe, device, **kwargs)
            end = time.perf_counter()
            if kwargs.get("mem_consumption"):
                memory_data_collector.stop_and_collect_data('from_pretrained_phase')
                memory_data_collector.log_data(compilation_phase=True)
            from_pretrain_time = end - start
        else:
            raise RuntimeError(f'==Failure ==: model path:{model_path} is not directory or directory is empty')
    else:
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not exist')

    log.info(f'Model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s')

    if device:
        # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
        if device.upper() == 'GPU':
            device = torch.device('cuda') if torch.cuda.is_available() else log.info('CUDA device is unavailable')
        else:
            device = torch.device(device.lower())
        log.info(f'Torch device was set to: {device}')

        pipe.to(device)
    else:
        raise RuntimeError('==Failure ==: no device to load')

    if kwargs['torch_compile_backend']:
        backend = kwargs['torch_compile_backend']
        compiled_model = run_torch_compile(pipe, backend, memory_data_collector if kwargs.get("mem_consumption") else None)
        pipe = compiled_model
    return pipe, from_pretrain_time, False, None


def create_text_2_speech_model(model_path, device, memory_data_collector, **kwargs):
    model_path = Path(model_path)
    from_pretrain_time = 0
    if model_path.exists():
        if model_path.is_dir() and len(os.listdir(model_path)) != 0:
            log.info(f'Load text to speech model from model path:{model_path}')
            model_class = kwargs['use_case'].pt_cls
            token_class = kwargs['use_case'].tokenizer_cls
            if kwargs.get("mem_consumption"):
                memory_data_collector.start()
            start = time.perf_counter()
            pipe = model_class.from_pretrained(model_path)
            vocoder = None
            if kwargs.get('vocoder_path'):
                vocoder = kwargs['use_case'].vocoder_cls
            pipe = set_bf16(pipe, device, **kwargs)
            end = time.perf_counter()
            if kwargs.get("mem_consumption"):
                memory_data_collector.stop_and_collect_data('from_pretrained_phase')
                memory_data_collector.log_data(compilation_phase=True)
            from_pretrain_time = end - start
            processor = token_class.from_pretrained(model_path)
        else:
            raise RuntimeError(f'==Failure ==: model path:{model_path} is not directory or directory is empty')
    else:
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not exist')

    log.info(f'Model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s')

    if device:
        # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
        if device.upper() == 'GPU':
            device = torch.device('cuda') if torch.cuda.is_available() else log.info('CUDA device is unavailable')
        else:
            device = torch.device(device.lower())
        log.info(f'Torch device was set to: {device}')

        pipe.to(device)
    else:
        raise RuntimeError('==Failure ==: no device to load')

    if kwargs['torch_compile_backend']:
        backend = kwargs['torch_compile_backend']
        compiled_model = run_torch_compile(pipe, backend, memory_data_collector if kwargs.get("mem_consumption") else None)
        pipe = compiled_model

    return pipe, processor, vocoder, from_pretrain_time, False


def create_ldm_super_resolution_model(model_path, device, memory_data_collector, **kwargs):
    model_path = Path(model_path)
    from_pretrain_time = 0
    if model_path.exists():
        if model_path.is_dir() and len(os.listdir(model_path)) != 0:
            log.info(f'Load image model from model path:{model_path}')
            model_class = kwargs['use_case'].pt_cls
            start = time.perf_counter()
            pipe = model_class.from_pretrained(model_path)
            end = time.perf_counter()
            if kwargs.get("mem_consumption"):
                memory_data_collector.stop_and_collect_data('from_pretrained_phase')
                memory_data_collector.log_data(compilation_phase=True)
            from_pretrain_time = end - start
        else:
            raise RuntimeError(f'==Failure ==: model path:{model_path} is not directory or directory is empty')
    else:
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not exist')

    log.info(f'Model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s')

    if device:
        # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
        if device.upper() == 'GPU':
            device = torch.device('cuda') if torch.cuda.is_available() else log.info('CUDA device is unavailable')
        else:
            device = torch.device(device.lower())
        log.info(f'Torch device was set to: {device}')

        pipe.to(device)
    else:
        raise RuntimeError('==Failure ==: no device to load')

    if kwargs['torch_compile_backend']:
        backend = kwargs['torch_compile_backend']
        compiled_model = run_torch_compile(pipe, backend, memory_data_collector if kwargs.get("mem_consumption") else None)
        pipe = compiled_model
    return pipe, from_pretrain_time


def create_text_reranker_model(model_path: Path, device: str, memory_monitor: MemMonitorWrapper, **kwargs):
    if not model_path.exists():
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not exist')
    if not device:
        raise RuntimeError('==Failure ==: no device to load')
    if not model_path.is_dir() or len(os.listdir(model_path)) == 0:
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not directory or directory is empty')

    log.info(f'Load text reranker model from model path:{model_path}')
    try:
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    except Exception:
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    rerank_use_case = kwargs['use_case']
    rerank_use_case.adjust_model_class_by_config(model_config)
    model_class = rerank_use_case.pt_cls
    token_class = rerank_use_case.tokenizer_cls
    if kwargs.get("mem_consumption"):
        memory_monitor.start()
    start = time.perf_counter()
    pipe = model_class.from_pretrained(model_path)
    pipe = set_bf16(pipe, device, **kwargs)
    end = time.perf_counter()
    if kwargs.get("mem_consumption"):
        memory_monitor.stop_and_collect_data('from_pretrained_phase')
        memory_monitor.log_data('for from pretrained phase')
    from_pretrain_time = end - start
    processor = token_class.from_pretrained(model_path)
    log.info(f'Model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s')

    # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
    if device.upper() == 'GPU':
        device = torch.device('cuda') if torch.cuda.is_available() else log.info('CUDA device is unavailable')
    else:
        device = torch.device(device.lower())
    log.info(f'Torch device was set to: {device}')

    pipe.to(device)

    if kwargs['torch_compile_backend']:
        backend = kwargs['torch_compile_backend']
        compiled_model = run_torch_compile(pipe, backend, memory_monitor if kwargs.get("mem_consumption") else None)
        pipe = compiled_model

    return pipe, processor, from_pretrain_time, None, False
