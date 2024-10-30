#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import openvino_genai
from PIL import Image
import numpy as np
import torch


class Generator(openvino_genai.Generator):
    def __init__(self, seed, mu=0.0, sigma=1.0):
        openvino_genai.Generator.__init__(self)
        self.mu = mu
        self.sigma = sigma
        self.generator = torch.Generator(device='cpu').manual_seed(seed)

    def next(self):
        return torch.randn(1, generator=self.generator).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well
    random_generator = Generator(42)  # openvino_genai.CppStdGenerator can be used to have same images as C++ sample
    # print(f'--- {random_generator.next()}')

    pipe = openvino_genai.Text2ImagePipeline(args.model_dir, device)
    image_tensor = pipe.generate(
        args.prompt,
        random_generator=random_generator
    )

    image = Image.fromarray(image_tensor.data[0])
    image.save("image.bmp")


if '__main__' == __name__:
    main()
