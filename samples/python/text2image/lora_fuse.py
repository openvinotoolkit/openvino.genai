#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import openvino as ov
import openvino_genai
import numpy as np
import sys


class Generator(openvino_genai.Generator):
    def __init__(self, seed, mu=0.0, sigma=1.0):
        openvino_genai.Generator.__init__(self)
        np.random.seed(seed)
        self.mu = mu
        self.sigma = sigma

    def next(self):
        return np.random.normal(self.mu, self.sigma)


def image_write(path: str, image_tensor: ov.Tensor):
    from PIL import Image
    image = Image.fromarray(image_tensor.data[0])
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models_path')
    parser.add_argument('prompt')
    args, adapters = parser.parse_known_args()

    prompt = args.prompt

    device = "CPU"  # GPU can be used as well

    # MODE_FUSE instructs pipeline to fuse adapter tensors into original model weights loaded into memory
    # giving the same performance level for inference as for the original model. After doing it you cannot
    # change adapter dynamically without re-initializing the pipeline from scratch.
    adapter_config = openvino_genai.AdapterConfig(openvino_genai.AdapterConfig.Mode.MODE_FUSE)

    # Multiple LoRA adapters applied simultaneously are supported, parse them all and corresponding alphas from cmd parameters:
    for i in range(int(len(adapters) / 2)):
        adapter = openvino_genai.Adapter(adapters[2 * i])
        alpha = float(adapters[2 * i + 1])
        adapter_config.add(adapter, alpha)

    # LoRA adapters passed to the constructor will be activated by default in next generates
    pipe = openvino_genai.Text2ImagePipeline(args.models_path, device, adapters=adapter_config)
    print("Generating image with LoRA adapters applied, resulting image will be in lora.bmp")
    image = pipe.generate(prompt,
                          generator=Generator(42),
                          width=512,
                          height=896,
                          num_inference_steps=20)

    image_write("lora.bmp", image)


if '__main__' == __name__:
    main()
