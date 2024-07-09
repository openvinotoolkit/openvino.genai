# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from functools import partial
import logging as log
from pathlib import Path
from datasets import load_dataset
import torch
import numpy as np
from nncf import compress_weights
from nncf import Dataset
from openvino import save_model
import nncf
from ..nncf_utils import COMPRESSION_OPTIONS, INT4_MODEL_CONFIGURATION
from optimum.intel.openvino.configuration import _check_default_4bit_configs
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


def transform_fn(item, item_name, input_shapes, tokenizer, config, max_tokens=127):
    tokenized_text = tokenizer(item[item_name], return_tensors="np")
    input_ids = tokenized_text["input_ids"][:max_tokens]
    attention_mask = tokenized_text["attention_mask"][:max_tokens]

    inputs = {}
    inputs["input_ids"] = input_ids

    if "attention_mask" in input_shapes:
        inputs["attention_mask"] = tokenized_text["attention_mask"]

    if "position_ids" in input_shapes:
        position_ids = np.cumsum(attention_mask, axis=1) - 1
        position_ids[attention_mask == 0] = 1
        inputs["position_ids"] = position_ids

    batch_size = input_ids.shape[0]
    if config.model_type == "bloom":
        batch_size *= config.num_attention_heads

    if "beam_idx" in input_shapes:
        inputs["beam_idx"] = np.arange(batch_size, dtype=int)

    for name, shape in input_shapes.items():
        if name in inputs:
            continue
        inputs[name] = np.zeros(shape)

    return inputs


def get_ov_input_shapes(model, batch_size=1):
    inputs = {}
    for val in model.inputs:
        name = val.any_name
        shape = list(val.partial_shape.get_min_shape())
        shape[0] = batch_size
        inputs[name] = shape

    return inputs


def get_data_aware_args(ov_model, tokenizer, config, compression_args, args):
    """initializes dict with data-aware compression parameters if defined dataset and tokenizer

    Args:
        ov_model : OpenVINO model for compression
        tokenizer : tokenizer for ov_model
        config : ov_model configuration
        compression_args: compression arguments from model compression configuration
        args : CLI args

    Returns:
        res: dict with data-aware compression parameters
    """
    res = {}
    if tokenizer is None:
        return res
    dataset_params = None
    if 'dataset' in compression_args:
        dataset_args = compression_args['dataset']
        dataset_params = dataset_args['name']
        if 'sensitivity_metric' in dataset_args:
            res['mode'] = dataset_args['sensitivity_metric']
        if 'awq' in dataset_args:
            res['awq'] = dataset_args['awq']
        if 'scale_estimation' in dataset_args:
            res['scale_estimation'] = dataset_args['scale_estimation']
    elif args.dataset is not None:
        dataset_params = args.dataset
        if args.awq:
            res['awq'] = args.awq
        if args.scale_estimation:
            res['scale_estimation'] = args.scale_estimation

    if dataset_params is not None:
        # for example "wikitext,wikitext-2-v1,train[:1000],text"
        path, name, split, item_name = dataset_params.split(',')
        dataset = load_dataset(path, name, split=split)

        if path == 'wikitext':
            # filter short sentences
            dataset = dataset.filter(lambda example: len(example["text"]) > 128)
        input_shapes = get_ov_input_shapes(ov_model)
        data_transform_func = partial(transform_fn, item_name=item_name, tokenizer=tokenizer, input_shapes=input_shapes, config=config)
        nncf_dataset = Dataset(dataset, data_transform_func)
        res['dataset'] = nncf_dataset
    return res


def compress_ov_model_weights_helper(ov_model, tok, config, out_path, compress_weights_format="INT8", fp16=False, args={}, model_name="openvino_model"):
    compression_args = None
    if "INT8" in compress_weights_format and "INT8_ASYM" in COMPRESSION_OPTIONS:
        warnings.warn("Usage INT8 mode is deprecated and will be removed soon. Please use INT8_ASYM instead", DeprecationWarning)
    if "4BIT_DEFAULT" in compress_weights_format:
        compression_args = _check_default_4bit_configs(config)
        if compression_args:
            sym = compression_args.pop("sym", False)
            compression_args.pop("bits", 4)
            compression_args["mode"] = nncf.CompressWeightsMode.INT4_SYM if sym else nncf.CompressWeightsMode.INT4_ASYM
        if compression_args is None:
            model_id = out_path.parents[3].name
            if model_id in INT4_MODEL_CONFIGURATION:
                compression_args = INT4_MODEL_CONFIGURATION[model_id]
            else:
                compression_args = COMPRESSION_OPTIONS["INT4_ASYM"]

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
    compression_args.update(get_data_aware_args(ov_model, tok, config, compression_args, args))
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
