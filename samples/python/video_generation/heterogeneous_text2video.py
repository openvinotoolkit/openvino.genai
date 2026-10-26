#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import openvino_genai

from video_utils import save_video


def main():
    parser = argparse.ArgumentParser(
        description="Generate LTX-Video with a separate device for the text encoder, denoiser, and VAE decoder"
    )
    parser.add_argument("model_dir", help="Path to the exported LTX-Video model directory")
    parser.add_argument("prompt", help="Text prompt for video generation")

    # Set devices to command-line args if specified, otherwise default to CPU.
    # Note that these can be set to CPU, GPU, or NPU.
    parser.add_argument("text_encoder_device", nargs="?", default="CPU", help="Text encoder device (default: CPU)")
    parser.add_argument("denoiser_device", nargs="?", default="CPU", help="Denoiser device (default: CPU)")
    parser.add_argument("vae_decoder_device", nargs="?", default="CPU", help="VAE decoder device (default: CPU)")
    parser.add_argument(
        "num_frames", type=int, nargs="?", default=161, help="Number of frames to generate (default: 161)"
    )

    args = parser.parse_args()

    width = 704
    height = 480
    num_frames = args.num_frames
    frame_rate = 25
    # Match the guidance scale used by the basic text2video sample.
    guidance_scale = 3.0
    seed = 42
    number_of_videos_to_generate = 1
    number_of_inference_steps_per_video = 25

    print(f"text_encoder_device = {args.text_encoder_device}")
    print(f"denoiser_device = {args.denoiser_device}")
    print(f"vae_decoder_device = {args.vae_decoder_device}")
    print(f"num_frames = {num_frames}")
    print(f"guidance_scale = {guidance_scale}")
    print(f"seed = {seed}")

    # Cache compiled models to speed up compilation on subsequent runs.
    ov_cache_dir = "./cache"

    #
    # Step 1: Create the initial Text2VideoPipeline, given the model path
    #
    pipe = openvino_genai.Text2VideoPipeline(args.model_dir)

    #
    # Step 2: Reshape the pipeline given number of videos, frames, height, width, and guidance scale.
    #
    pipe.reshape(number_of_videos_to_generate, num_frames, height, width, guidance_scale)

    #
    # Step 3: Compile the pipeline given the specified devices, and properties (like cache dir)
    #
    properties = {"CACHE_DIR": ov_cache_dir}

    # Note that if there are device-specific properties that are needed, they can
    # be added using a "DEVICE_PROPERTIES" entry, like this:
    # properties = {
    #    "DEVICE_PROPERTIES":
    #    {
    #        "CPU": {"CACHE_DIR": "cpu_cache"},
    #        "GPU": {"CACHE_DIR": "gpu_cache"},
    #        "NPU": {"CACHE_DIR": "npu_cache"}
    #    }
    # }

    pipe.compile(args.text_encoder_device, args.denoiser_device, args.vae_decoder_device, config=properties)

    #
    # Step 4: Use the Text2VideoPipeline to generate 'number_of_videos_to_generate' videos.
    #

    generator = openvino_genai.CppStdGenerator(seed)
    for i in range(0, number_of_videos_to_generate):
        output = pipe.generate(
            args.prompt,
            negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
            num_inference_steps=number_of_inference_steps_per_video,
            generator=generator,
            frame_rate=frame_rate,
        )

        save_video("video_" + str(i) + ".avi", output.video, frame_rate)


if __name__ == "__main__":
    main()
