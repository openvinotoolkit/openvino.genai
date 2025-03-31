#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from openvino import Tensor
import openvino_genai
import numpy as np


def decrypt_model(model_dir, model_file_name, weights_file_name):
    with open(model_dir + '/' + model_file_name, "r") as file:
        model = file.read()
    # decrypt model

    with open(model_dir + '/' + weights_file_name, "rb") as file:
        binary_data = file.read()
    # decrypt weights
    weights = np.frombuffer(binary_data, dtype=np.uint8).astype(np.uint8)

    return model, Tensor(weights)

def read_tokenizer(model_dir):
    tokenizer_model_name = 'openvino_tokenizer.xml'
    tokenizer_weights_name = 'openvino_tokenizer.bin'
    tokenizer_model, tokenizer_weights = decrypt_model(model_dir, tokenizer_model_name, tokenizer_weights_name)

    detokenizer_model_name = 'openvino_detokenizer.xml'
    detokenizer_weights_name = 'openvino_detokenizer.bin'
    detokenizer_model, detokenizer_weights = decrypt_model(model_dir, detokenizer_model_name, detokenizer_weights_name)

    return openvino_genai.Tokenizer(tokenizer_model, tokenizer_weights, detokenizer_model, detokenizer_weights)

import base64

def encrypt_base64(src: bytes):
    return base64.b64encode(src)

def decrypt_base64(src: bytes):
    return base64.b64decode(src)

def get_config_for_cache_encryption(cache_dir, is_gpu):
    config_cache = dict()
    config_cache["CACHE_DIR"] = cache_dir
    config_cache["CACHE_ENCRYPTION_CALLBACKS"] = [encrypt_base64, decrypt_base64]
    if is_gpu:
        # set CACHE_MODE to OPTIMIZE_SIZE only for GPU to enable weightless cache
        config_cache["CACHE_MODE"] = "OPTIMIZE_SIZE"
    return config_cache

def main():
    parser = argparse.ArgumentParser(description="Help command")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to model and tokenizers base directory")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Prompt")
    parser.add_argument("-d", "--device", type=str, default="CPU", help="Device")
    parser.add_argument("-c", "--cache_dir", type=str, help="Path to cache dir (optional)")
    args = parser.parse_args()

    model, weights = decrypt_model(args.model, 'openvino_model.xml', 'openvino_model.bin')
    tokenizer = read_tokenizer(args.model)

    device = args.device
    config = dict()
    if args.cache_dir is not None:
        config = get_config_for_cache_encryption(args.cache_dir, device == 'GPU')

    pipe = openvino_genai.LLMPipeline(model, weights, tokenizer, device, {}, **config)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    print(pipe.generate(args.prompt, config))

if '__main__' == __name__:
    main()
