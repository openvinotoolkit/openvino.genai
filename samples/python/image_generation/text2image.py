#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import openvino_genai
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well
    pipe = openvino_genai.Text2ImagePipeline(args.model_dir, device)

    def callback(step, num_steps, latent):
        print(f"Step {step + 1}/{num_steps}")
        return False

    image_tensor = pipe.generate(
        args.prompt,
        width=512,
        height=512,
        num_inference_steps=20,
        num_images_per_prompt=1,
        callback=callback)

    image = Image.fromarray(image_tensor.data[0])
    image.save("image.bmp")


if __name__ == '__main__':
    main()