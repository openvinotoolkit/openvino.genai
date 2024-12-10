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
    guidance_scale = 7.5
    number_of_images_to_generate = 1
    number_of_inference_steps_per_image = 20

    print(f"text_encoder_device = {args.text_encoder_device}")
    print(f"unet_device = {args.unet_device}")
    print(f"vae_decoder_device = {args.vae_decoder_device}")

    # this is the path to where compiled models will get cached
    # (so that the 'compile' method run much faster 2nd+ time)
    ov_cache_dir = "./cache"

    #
    # Step 1: Prepare each Text2Image subcomponent (scheduler, text encoder, unet, vae) separately.
    #

    # Create the scheduler from the details listed in the json.
    scheduler = openvino_genai.Scheduler.from_config(args.model_dir + "/scheduler/scheduler_config.json")

    # Note that we can also create the scheduler by specifying specific type (for example EULER_DISCRETE), like this:
    # scheduler = openvino_genai.Scheduler.from_config(args.model_dir + "/scheduler/scheduler_config.json",
    #                                                  openvino_genai.Scheduler.Type.EULER_DISCRETE)
    # This can be useful when a particular type of Scheduler is not yet supported natively by OpenVINO GenAI.
    # (even though we are actively working to support most commonly used ones)

    # Create unet object
    unet = openvino_genai.UNet2DConditionModel(args.model_dir + "/unet")

    # Set batch size based on classifier free guidance condition.
    unet_batch_size = 2 if unet.do_classifier_free_guidance(guidance_scale) else 1

    # Create the text encoder
    text_encoder = openvino_genai.CLIPTextModel(args.model_dir + "/text_encoder")

    # In case of NPU, we need to reshape the model to have static shapes
    if args.text_encoder_device == "NPU":
        text_encoder.reshape(unet_batch_size)

    # Compile text encoder for the specified device
    text_encoder.compile(args.text_encoder_device, CACHE_DIR=ov_cache_dir)

    # In case of NPU, we need to reshape the unet model to have static shapes
    if args.unet_device == "NPU":
        # The max_postion_embeddings config from text encoder will be used as a parameter to unet reshape.
        max_position_embeddings = text_encoder.get_config().max_position_embeddings

        unet.reshape(unet_batch_size, height, width, max_position_embeddings)

    # Compile unet for specified device
    unet.compile(args.unet_device, CACHE_DIR=ov_cache_dir)

    # Create the decoder
    vae = openvino_genai.AutoencoderKL(args.model_dir + "/vae_decoder")

    # In case of NPU, we need to reshape the vae model to have static shapes
    if args.vae_decoder_device == "NPU":
        vae.reshape(1, height, width)

    # Compile vae decoder for the specified device
    vae.compile(args.vae_decoder_device, CACHE_DIR=ov_cache_dir)

    #
    # Step 2: Create a Text2ImagePipeline from the individual subcomponents
    #

    pipe = openvino_genai.Text2ImagePipeline.stable_diffusion(scheduler, text_encoder, unet, vae)

    #
    # Step 3: Use the Text2ImagePipeline to generate 'number_of_images_to_generate' images.
    #

    for imagei in range(0, number_of_images_to_generate):
        image_tensor = pipe.generate(
            args.prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=number_of_inference_steps_per_image,
            num_images_per_prompt=1,
            generator=openvino_genai.CppStdGenerator(42)
        )

        image = Image.fromarray(image_tensor.data[0])
        image.save("image_" + str(imagei) + ".bmp")


if '__main__' == __name__:
    main()
