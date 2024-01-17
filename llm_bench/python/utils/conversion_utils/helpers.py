# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from pathlib import Path
import logging as log
from typing import Union

import torch
from nncf import compress_weights
from openvino import save_model
from openvino.runtime.exceptions import OVTypeError
from ..nncf_utils import COMPRESSION_OPTIONS, INT4_MODEL_CONFIGURATION
import warnings


class BackendType(Enum):
    PYTORCH = 'pytorch'
    OPENVINO = 'openvino'


PYTORCH_DIR = 'pytorch'
PYTORCH_COMPRESS_WEIGHTS_DIR = 'compressed_weights/PT_{precision}-{compression}'
OV_DIR = 'dldt'
GPTQ_DIR = "GPTQ_INT4-{precision}"
OV_TOKENIZER_NAME = "openvino_tokenizer.xml"
OV_DETOKENIZER_NAME = "openvino_detokenizer.xml"


def is_torch_compression(args):
    return args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends


def is_ov_compression(args):
    return args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends


def is_fp16(args):
    return args.precision == "FP16"


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


# TODO: Replace with function from optimum-intel
def save_ov_tokenizer(
    tokenizer,
    output_path: Union[str, Path],
) -> None:
    from transformers import T5Tokenizer, T5TokenizerFast

    UNSUPPORTED_TOKENZIER_CLASSES = (
        T5Tokenizer,
        T5TokenizerFast,
    )
    if isinstance(tokenizer, UNSUPPORTED_TOKENZIER_CLASSES):
        log.info("OpenVINO Tokenizer for this model is not supported.")
        return

    try:
        from openvino_tokenizers import convert_tokenizer
    except ModuleNotFoundError:
        log.info("Run `pip install openvino-tokenizers` to get OpenVINO tokenizer/detokenizer models.")

    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    try:
        converted = convert_tokenizer(tokenizer, with_detokenizer=True)
    except NotImplementedError:
        log.info("Detokenizer is not supported, convert tokenizer only.")
        converted = convert_tokenizer(tokenizer, with_detokenizer=False)
    except OVTypeError:
        log.info("OpenVINO Tokenizer for this model is not supported.")
        return
    except Exception as exception:
        log.warning(f"OpenVINO Tokenizer for this model is not supported. Exception: {exception}")
        return

    if not isinstance(converted, tuple):
        converted = (converted,)

    for model, file_name in zip(converted, (OV_TOKENIZER_NAME, OV_DETOKENIZER_NAME)):
        save_model(model, output_path / file_name)


def save_tokenizer(tokenizer, out_dir: str, add_ov_tokenizer: bool = False) -> None:
    try:
        tokenizer.save_pretrained(out_dir)
    except Exception as e:
        log.error(f"Huggingface tokenizer saving failed with {e}")

    if add_ov_tokenizer:
        try:
            save_ov_tokenizer(tokenizer, out_dir)
        except Exception as e:
            log.error(f"OpenVINO tokenizer saving failed with {e}")

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
