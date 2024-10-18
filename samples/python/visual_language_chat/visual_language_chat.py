#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino_genai
from PIL import Image
from openvino import Tensor


def streamer(subword: str) -> bool:
    '''

    Args:
        subword: sub-word of the generated text.

    Returns: Return flag corresponds whether generation should be stopped.

    '''
    print(subword, end='', flush=True)

    # No value is returned as in this example we don't want to stop the generation in this method.
    # "return None" will be treated the same as "return False".


def read_image(path: str) -> Tensor:
    '''

    Args:
        path: The path to the image.

    Returns: the ov.Tensor containing the image.

    '''
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic.getdata()).reshape(1, 3, pic.size[1], pic.size[0]).astype(np.byte)
    return Tensor(image_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('image_dir')
    args = parser.parse_args()

    image = read_image(args.image_dir)

    device = 'CPU'  # GPU can be used as well
    enable_compile_cache = dict()
    if "GPU" == device:
        # Cache compiled models on disk for GPU to save time on the
        # next run. It's not beneficial for CPU.
        enable_compile_cache["CACHE_DIR"] = "vlm_cache"
    pipe = openvino_genai.VLMPipeline(args.model_dir, device, enable_compile_cache)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    pipe.start_chat()
    prompt = input('question:\n')
    pipe.generate(prompt, image=image, generation_config=config, streamer=streamer)

    while True:
        try:
            prompt = input("\n----------\n"
                "question:\n")
        except EOFError:
            break
        pipe.generate(prompt, generation_config=config, streamer=streamer)
    pipe.finish_chat()


if '__main__' == __name__:
    main()
