#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino_genai
import openvino
from PIL import Image
from openvino import Tensor
from pathlib import Path


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


def streamer(subword: str) -> bool:
    '''

    Args:
        subword: sub-word of the generated text.

    Returns: Return flag corresponds whether generation should be stopped.

    '''
    print(subword, end='', flush=True)

    # No value is returned as in this example we don't want to stop the generation in this method.
    # "return None" will be treated the same as "return openvino_genai.StreamingStatus.RUNNING".


def read_image(path: str) -> Tensor:
    '''

    Args:
        path: The path to the image.

    Returns: the ov.Tensor containing the image.

    '''
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic)
    return Tensor(image_data)


def read_images(path: str) -> list[Tensor]:
    entry = Path(path)
    if entry.is_dir():
        return [read_image(str(file)) for file in sorted(entry.iterdir())]
    return [read_image(path)]


# here is example how to make cache de-encryption based on base64
import base64

def encrypt_base64(src: bytes):
    return base64.b64encode(src)


def decrypt_base64(src: bytes):
    return base64.b64decode(src)


def get_config_for_cache_encryption():
    config_cache = dict()
    config_cache["CACHE_DIR"] = "llm_cache"
    config_cache["CACHE_ENCRYPTION_CALLBACKS"] = [encrypt_base64, decrypt_base64]
    config_cache["CACHE_MODE"] = "OPTIMIZE_SIZE"
    return config_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('image_dir', help="Image file or dir with images")
    parser.add_argument('prompt', help="Image file or dir with images")
    args = parser.parse_args()

    model_name_to_file_map = {
        ('language', 'openvino_language_model'),
        ('resampler', 'openvino_resampler_model'),
        ('text_embeddings', 'openvino_text_embeddings_model'),
        ('vision_embeddings', 'openvino_vision_embeddings_model')}

    models_map = dict()
    for model_name, file_name in model_name_to_file_map:
        model, weights = decrypt_model(args.model_dir, file_name + '.xml', file_name + '.bin')
        models_map[model_name] = (model, weights)

    tokenizer = read_tokenizer(args.model_dir)

    # GPU can be used as well.
    device = 'CPU'
    enable_compile_cache = dict()
    if "GPU" == device:
        # Cache compiled models on disk for GPU to save time on the
        # next run. It's not beneficial for CPU.
        enable_compile_cache = get_config_for_cache_encryption()

    pipe = openvino_genai.VLMPipeline(models_map, tokenizer, args.model_dir, device, **enable_compile_cache)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    rgbs = read_images(args.image_dir)

    pipe.generate(args.prompt, images=rgbs, generation_config=config, streamer=streamer)


if '__main__' == __name__:
    main()
