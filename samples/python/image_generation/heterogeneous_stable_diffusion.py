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

    # Set devices to command-line args if specified, otherwise default to CPU.
    # Note that these can be set to CPU, GPU, or NPU.
    parser.add_argument('text_encoder_device', nargs='?', default='CPU')
    parser.add_argument('unet_device', nargs='?', default='CPU')
    parser.add_argument('vae_decoder_device', nargs='?', default='CPU')

    args = parser.parse_args()

    width = 512
    height = 512
    number_of_images_to_generate = 1
    number_of_inference_steps_per_image = 20

    print(f"text_encoder_device = {args.text_encoder_device}")
    print(f"unet_device = {args.unet_device}")
    print(f"vae_decoder_device = {args.vae_decoder_device}")

    # this is the path to where compiled models will get cached
    # (so that the 'compile' method run much faster 2nd+ time)
    ov_cache_dir = "./cache"

    #
    # Step 1: Create the initial Text2ImagePipeline, given the model path
    #
    pipe = openvino_genai.Text2ImagePipeline(args.model_dir)

    #
    # Step 2: Reshape the pipeline given number of images, height, width, and guidance scale.
    #
    pipe.reshape(1, height, width, pipe.get_generation_config().guidance_scale)

    #
    # Step 3: Compile the pipeline given the specified devices, and properties (like cache dir)
    #
    properties = {"CACHE_DIR": ov_cache_dir}

    # Note that if there are device-specific properties that are needed, they can
    # be added using a "DEVICE_PROPERTIES" entry, like this:
    #properties = {
    #    "DEVICE_PROPERTIES":
    #    {
    #        "CPU": {"CACHE_DIR": "cpu_cache"},
    #        "GPU": {"CACHE_DIR": "gpu_cache"},
    #        "NPU": {"CACHE_DIR": "npu_cache"}
    #    }
    #}

    pipe.compile(args.text_encoder_device, args.unet_device, args.vae_decoder_device, config=properties)

    #
    # Step 4: Use the Text2ImagePipeline to generate 'number_of_images_to_generate' images.
    #

    for imagei in range(0, number_of_images_to_generate):
        image_tensor = pipe.generate(
            args.prompt,
            num_inference_steps=number_of_inference_steps_per_image,
        )

        image = Image.fromarray(image_tensor.data[0])
        image.save("image_" + str(imagei) + ".bmp")


if '__main__' == __name__:
    main()
