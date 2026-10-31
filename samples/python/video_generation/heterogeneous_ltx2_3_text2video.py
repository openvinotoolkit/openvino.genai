#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# LTX-2.3 text-to-video with the pipeline stages split across devices.
#
# Export the model first:
#   optimum-cli export openvino --model diffusers/LTX-2.3-Diffusers \
#       --task text-to-video --weight-format int8 --trust-remote-code ltx2_3_int8
#
# Then run the transformer on GPU and everything else on CPU:
#   python heterogeneous_ltx2_3_text2video.py ltx2_3_int8 "a robot dancing in the rain" CPU GPU CPU

import argparse

import openvino_genai
from video_utils import save_audio, save_video

DEFAULT_PROMPT = "A medium shot of a young girl in a bright yellow summer dress jumping into a water puddle on a wet asphalt street. Heavy rain falls continuously, creating ripples on the water surface and splash effects as her feet land. Cinematic lighting, ultra-realistic textures, natural movement, high detail, 4k resolution. Sound of heavy raindrops falling on asphalt, clear splash sound of water when she jumps, quiet wind rustling, crisp natural ambient noise. Absolute silence for background music, no music, no soundtrack."
DEFAULT_NEGATIVE_PROMPT = "blurry, distorted face, low quality"

WIDTH = 1280
HEIGHT = 704
NUM_FRAMES = 49
FRAME_RATE = 24.0
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 3.0
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)

    # compile() takes three devices, not one per model. LTX-2.3 has seven models and they are
    # grouped as follows, so e.g. the connectors always follow the text encoder:
    #   text_encode_device -> text_encoder, connectors
    #   denoise_device     -> transformer
    #   vae_device         -> vae, audio_vae, vocoder
    parser.add_argument("text_encode_device", nargs="?", default="CPU", help="text encoder and connectors")
    parser.add_argument("denoise_device", nargs="?", default="GPU", help="transformer (the denoiser)")
    parser.add_argument("vae_device", nargs="?", default="CPU", help="video VAE, audio VAE and vocoder")

    parser.add_argument("--cache-dir", default="./cache", help="where compiled models are cached")
    parser.add_argument(
        "--inference-precision",
        default="f16",
        choices=["f16", "f32"],
        help="floating-point compute precision (INFERENCE_PRECISION_HINT) of the denoise device only",
    )
    parser.add_argument("--output-prefix", default="genai", help="prefix for the output video/audio files")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument(
        "--reshape",
        action="store_true",
        help="compile for the selected static generation shapes",
    )
    args = parser.parse_args()

    print(f"text_encode_device = {args.text_encode_device} (text_encoder, connectors)")
    print(f"denoise_device     = {args.denoise_device} (transformer)")
    print(f"vae_device         = {args.vae_device} (vae, audio_vae, vocoder)")

    # Constructing without a device reads the models but does not compile them. That is what
    # leaves reshape() and the three-device compile() available; passing a device here would
    # compile everything onto that one device immediately.
    pipe = openvino_genai.Text2VideoPipeline(args.model_dir)

    gen_config = pipe.get_generation_config()
    gen_config.negative_prompt = DEFAULT_NEGATIVE_PROMPT
    gen_config.width = args.width
    gen_config.height = args.height
    gen_config.num_frames = args.num_frames
    gen_config.frame_rate = FRAME_RATE
    gen_config.num_inference_steps = args.num_inference_steps
    gen_config.guidance_scale = GUIDANCE_SCALE
    gen_config.stg_scale = 1.0
    gen_config.audio_stg_scale = 1.0
    gen_config.modality_scale = 3.0
    gen_config.audio_modality_scale = 3.0
    gen_config.spatio_temporal_guidance_blocks = [28]
    gen_config.generator = openvino_genai.TorchGenerator(SEED)
    pipe.set_generation_config(gen_config)

    if args.reshape:
        # Has to happen before compile() — a compiled pipeline cannot be reshaped. Static shapes
        # are worth more on GPU than on CPU, since they avoid recompiling per shape.
        pipe.reshape(
            gen_config.num_videos_per_prompt,
            gen_config.num_frames,
            gen_config.height,
            gen_config.width,
            gen_config.guidance_scale,
        )

    properties = {
        "CACHE_DIR": args.cache_dir,
        "DEVICE_PROPERTIES": {args.denoise_device: {"INFERENCE_PRECISION_HINT": args.inference_precision}},
    }

    # If a device needs its own settings, pass them per device instead:
    # properties = {
    #     "DEVICE_PROPERTIES": {
    #         "CPU": {"CACHE_DIR": "cpu_cache"},
    #         "GPU": {"CACHE_DIR": "gpu_cache"},
    #     }
    # }

    pipe.compile(args.text_encode_device, args.denoise_device, args.vae_device, config=properties)

    output = pipe.generate(args.prompt)

    save_video(f"{args.output_prefix}_video.avi", output.video, int(gen_config.frame_rate))
    save_audio(f"{args.output_prefix}_audio.wav", output.audio, output.audio_sample_rate)


if __name__ == "__main__":
    main()
