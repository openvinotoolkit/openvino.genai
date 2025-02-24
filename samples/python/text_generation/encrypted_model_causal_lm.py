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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')
    args = parser.parse_args()

    model, weights = decrypt_model(args.model_dir, 'openvino_model.xml', 'openvino_model.bin')
    tokenizer = read_tokenizer(args.model_dir)

    device = 'CPU'  # GPU can be used as well
    pipe = openvino_genai.LLMPipeline(model, weights, tokenizer, device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    print(pipe.generate(args.prompt, config))


if '__main__' == __name__:
    main()
