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
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well
    pipe = openvino_genai.Image2ImagePipeline(args.model_dir, device)

    image = read_image(args.image)

    def callback(step, num_steps, latent):
        print(f"Step {step + 1}/{num_steps}")
        return False

    image_tensor = pipe.generate(
        args.prompt,
        image,
        strength=0.8,
        callback=callback
    )

    image = Image.fromarray(image_tensor.data[0])
    image.save("image.bmp")


if __name__ == '__main__':
    main()
