#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino
import openvino_genai
import torch
import numpy as np

from PIL import Image

class Generator(openvino_genai.Generator):
    def __init__(self, seed):
        openvino_genai.Generator.__init__(self)
        self.generator = torch.Generator(device='cpu').manual_seed(seed)

    def next(self):
        return torch.randn(1, generator=self.generator, dtype=torch.float32).item()

    def randn_tensor(self, shape: openvino.Shape):
        torch_tensor = torch.randn(list(shape), generator=self.generator, dtype=torch.float32)
        return openvino.Tensor(torch_tensor.numpy())


def read_image(path: str) -> openvino.Tensor:
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic.getdata()).reshape(1, pic.size[1], pic.size[0], 3).astype(np.uint8)
    return openvino.Tensor(image_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')
    parser.add_argument('image')
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well
    pipe = openvino_genai.Image2ImagePipeline(args.model_dir, device)

    initial_image = read_image(args.image)
    _, H, W, _ = list(initial_image.shape)

    image_tensor = pipe.generate(
        args.prompt,
        read_image(args.image),
        width=W,
        height=H,
        strength=0.8,
        generator=Generator(42)
    )

    image = Image.fromarray(image_tensor.data[0])
    image.save("/home/devuser/ilavreno/openvino.genai/genai_image.bmp")


if '__main__' == __name__:
    main()
