# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from pathlib import Path
import logging as log
import torch
from nncf import compress_weights
from openvino import save_model
from ..nncf_utils import COMPRESSION_OPTIONS, INT4_MODEL_CONFIGURATION
import warnings


class BackendType(Enum):
    PYTORCH = 'pytorch'
    OPENVINO = 'openvino'


PYTORCH_DIR = 'pytorch'
PYTORCH_COMPRESS_WEIGHTS_DIR = 'compressed_weights/PT_{precision}-{compression}'
OV_DIR = 'dldt'
GPTQ_DIR = "GPTQ_INT4-{precision}"


def is_torch_compression(args):
    return args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends


def is_ov_compression(args):
    return args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends


def is_fp16(args):
    return args.precision == "FP16"


def is_int8_compression(compress_weights_mode):
    return compress_weights_mode in ["INT8", "INT8_ASYM", "INT8_SYM"]


def is_ov_model_provided(model_id, model_dir, precision, model_name="openvino_model.xml"):
    model_dirs = []
    if Path(model_id).is_dir():
        model_dirs.append(Path(model_id))
        model_dirs.append(Path(model_id) / precision)
        model_dirs.append(Path(model_id) / OV_DIR / precision)
        model_dirs.append(Path(model_id) / PYTORCH_DIR / OV_DIR / precision)
    model_dir = Path(model_dir)
    model_dirs.append(model_dir)
    model_dirs.append(model_dir / precision)
    model_dirs.append(model_dir / OV_DIR / precision)
    model_dirs.append(model_dir / PYTORCH_DIR / OV_DIR / precision)
    for md in model_dirs:
        found = True
        for suffix in ['.xml', '.bin']:
            model_file = (md / model_name).with_suffix(suffix)
            if not model_file.exists():
                found = False
                break
        if found:
            return found
    return False


def get_fp_path(args, model_subpath):
    model_dirs = []
    if Path(args.model_id).is_dir():
        base_model_dir = Path(args.model_id)
        model_dirs.extend([
            base_model_dir, base_model_dir / args.precision, base_model_dir / OV_DIR / args.precision, base_model_dir / PYTORCH_DIR / OV_DIR / args.precision
        ])
    model_dir = Path(args.output_dir)
    model_dirs.append(model_dir)
    model_dirs.append(Path(model_dir) / args.precision)
    model_dirs.append(Path(model_dir) / OV_DIR / args.precision)
    model_dirs.append(Path(model_dir) / PYTORCH_DIR / OV_DIR / args.precision)
    for md in model_dirs:
        if (md / model_subpath).exists():
            return md / model_subpath
    return None


def save_tokenizer(tokenizer, out_dir):
    try:
        tokenizer.save_pretrained(out_dir)
    except Exception as e:
        log.error(f'tokenizer loading failed with {e}')


def compress_ov_model_weights_helper(ov_model, tok, config, out_path, compress_weights_format="INT8", fp16=False, args={}, model_name="openvino_model"):
    compression_args = None
    if "INT8" in compress_weights_format and "INT8_ASYM" in COMPRESSION_OPTIONS:
        warnings.warn("Usage INT8 mode is deprecated and will be removed soon. Please use INT8_ASYM instead", DeprecationWarning)
    if "4BIT_DEFAULT" in compress_weights_format:
        model_id = out_path.parents[3].name
        if model_id in INT4_MODEL_CONFIGURATION:
            compression_args = INT4_MODEL_CONFIGURATION[model_id]
        else:
            compression_args = COMPRESSION_OPTIONS["INT4_SYM"]

    if compression_args is None:
        compression_args = COMPRESSION_OPTIONS[compress_weights_format]
        if args.ratio is not None:
            compression_args["ratio"] = args.ratio
        if args.group_size is not None:
            compression_args["group_size"] = args.group_size
    if args.all_layers:
        compression_args["all_layers"] = True
    log.info("Compression options:")
    log.info(compression_args)
    compressed_ov_model = compress_weights(ov_model, **compression_args)
    save_ov_model_helper(compressed_ov_model, out_path, model_name, fp16=fp16, tok=tok, config=config)


def save_ov_model_helper(ov_model, out_path, model_name='openvino_model', fp16=False, tok=None, config=None):
    model_name = model_name or "openvino_model"
    save_model(ov_model, Path(out_path) / f'{model_name}.xml', compress_to_fp16=fp16)
    if tok is not None:
        save_tokenizer(tok, out_path)
    if config is not None:
        config.save_pretrained(out_path)


def is_gptq(config):
    config_dict = config.to_dict()
    quantization_config = config_dict.get("quantization_config", None)
    return quantization_config and quantization_config["quant_method"] == "gptq"


def flattenize_inputs(inputs):
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs


def patch_gptq(config):
    do_gptq_patching = False
    do_gptq_patching = is_gptq(config)
    orig_cuda_check = torch.cuda.is_available
    orig_post_init_model = None
    if do_gptq_patching:
        torch.set_default_dtype(torch.float32)
        torch.cuda.is_available = lambda: True

        from optimum.gptq import GPTQQuantizer

        orig_post_init_model = GPTQQuantizer.post_init_model

        def post_init_model(self, model):
            from auto_gptq import exllama_set_max_input_length

            class StoreAttr(object):
                pass

            model.quantize_config = StoreAttr()
            model.quantize_config.desc_act = self.desc_act
            if self.desc_act and not self.disable_exllama and self.max_input_length is not None:
                model = exllama_set_max_input_length(model, self.max_input_length)
            return model

        GPTQQuantizer.post_init_model = post_init_model
    return orig_cuda_check, orig_post_init_model


def unpatch_gptq(orig_cuda_check, orig_post_init_model):
    from optimum.gptq import GPTQQuantizer
    torch.cuda.is_available = orig_cuda_check
    GPTQQuantizer.post_init_model = orig_post_init_model
