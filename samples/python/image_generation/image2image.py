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
    parser.add_argument('image', nargs='?', default=None)
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well
    pipe = openvino_genai.Image2ImagePipeline(args.model_dir, device)

    if args.image is None:
        image = read_image(args.image)
        image_tensor = pipe.generate(args.prompt, image,
            strength=0.8 # controls how initial image is noised after being converted to latent space. `1` means initial image is fully noised
        )
    else:
        image_tensor = pipe.generate(
            args.prompt,
            width=512,
            height=512,
            num_inference_steps=20,
            num_images_per_prompt=1,
            strength=1.0 # 'strength' must be 1.0f for Image 2 image pipeline without initial image
        )

    image = Image.fromarray(image_tensor.data[0])
    image.save("image.bmp")


if '__main__' == __name__:
    main()
