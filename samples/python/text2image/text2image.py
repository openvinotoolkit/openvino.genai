#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import random

import openvino_genai
from PIL import Image


class Generator(openvino_genai.Generator):
    def __init__(self, seed):
        openvino_genai.Generator.__init__(self)
        random.seed(seed)

    def next(self):
        return random.random()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well
    random_generator = Generator(50)  # openvino_genai.CppStdGenerator can be used
    pipe = openvino_genai.Text2ImagePipeline(args.model_dir, device)
    image_tensor = pipe.generate(
        args.prompt,
        width=512,
        height=512,
        num_inference_steps=20,
        num_images_per_prompt=1,
        random_generator=random_generator,
    )

    image = Image.fromarray(image_tensor.data[0])
    image.save("image.bmp")


if '__main__' == __name__:
    main()
