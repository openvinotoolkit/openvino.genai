#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino
import openvino_genai
import numpy as np

from PIL import Image

def read_image(path: str) -> openvino.Tensor:
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic)[None]
    return openvino.Tensor(image_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')
    parser.add_argument('image')
    parser.add_argument('mask')
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well
    pipe = openvino_genai.InpaintingPipeline(args.model_dir, device)

    image = read_image(args.image)
    mask_image = read_image(args.mask)

    image_tensor = pipe.generate(args.prompt, image, mask_image)

    image = Image.fromarray(image_tensor.data[0])
    image.save("image.bmp")


if '__main__' == __name__:
    main()
