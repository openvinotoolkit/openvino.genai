#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import openvino_genai

from pathlib import Path

from PIL import Image


def pipeline_export_import(root_dir: Path):
    pipe = openvino_genai.Text2ImagePipeline(root_dir, "CPU")
    pipe.export_model(root_dir / "exported")
    # pipeline models are exported to dedicated subfolders
    # for stable diffusion xl:
    # exported/
    # ├── text_encoder/
    # │   └── openvino_model.blob
    # ├── text_encoder_2/
    # │   └── openvino_model.blob
    # ├── unet/
    # │   └── openvino_model.blob
    # └── vae_decoder/
    #     └── openvino_model.blob

    # during import, specify blob_path property to point to the exported model location
    imported_pipe = openvino_genai.Text2ImagePipeline(root_dir, "CPU", blob_path=root_dir / "exported")


def dedicated_models_export_import(root_dir: Path):
    blob_path = root_dir / "blobs"
    device = "CPU"

    text_encoder = openvino_genai.CLIPTextModel(root_dir / "text_encoder", device)
    text_encoder.export_model(blob_path / "text_encoder")

    text_encoder_2 = openvino_genai.CLIPTextModelWithProjection(root_dir / "text_encoder_2", device)
    text_encoder_2.export_model(blob_path / "text_encoder_2")

    unet = openvino_genai.UNet2DConditionModel(root_dir / "unet", device)
    unet.export_model(blob_path / "unet")

    vae = openvino_genai.AutoencoderKL(root_dir / "vae_decoder")
    vae.compile(device)
    vae.export_model(blob_path)
    # AutoencoderKL can be composed with decoder and encoder models
    # exported/
    # └── vae_decoder/
    #     └── openvino_model.blob
    # └── vae_encoder/
    #     └── openvino_model.blob

    pipe = openvino_genai.Text2ImagePipeline.stable_diffusion_xl(
        scheduler=openvino_genai.Scheduler.from_config(root_dir / "scheduler" / "scheduler_config.json"),
        clip_text_model=openvino_genai.CLIPTextModel(root_dir / "text_encoder", device, blob_path=blob_path / "text_encoder"),
        clip_text_model_with_projection=openvino_genai.CLIPTextModelWithProjection(root_dir / "text_encoder_2", device, blob_path=blob_path / "text_encoder_2"),
        unet=openvino_genai.UNet2DConditionModel(root_dir / "unet", device, blob_path=blob_path / "unet"),
        vae=openvino_genai.AutoencoderKL(root_dir / "vae_decoder", device, blob_path=blob_path),
    )


def export_import_with_reshape(root_dir: Path, prompt: str):
    device = "CPU"

    width = 512
    height = 512
    number_of_images_to_generate = 1
    number_of_inference_steps_per_image = 20

    pipe = openvino_genai.Text2ImagePipeline(root_dir)
    pipe.reshape(1, height, width, pipe.get_generation_config().guidance_scale)
    pipe.compile(device)
    pipe.export_model(root_dir / "exported")

    imported_pipe = openvino_genai.Text2ImagePipeline(root_dir, device, blob_path=root_dir / "exported")

    # update generation config according to the new shape parameters
    config = imported_pipe.get_generation_config()
    config.height = height
    config.width = width
    config.num_images_per_prompt = number_of_images_to_generate
    imported_pipe.set_generation_config(config)

    for imagei in range(0, number_of_images_to_generate):
        image_tensor = imported_pipe.generate(
            prompt,
            num_inference_steps=number_of_inference_steps_per_image,
        )

        image = Image.fromarray(image_tensor.data[0])
        image.save("image_" + str(imagei) + ".bmp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("prompt")

    args = parser.parse_args()

    root_dir = Path(args.model_dir)

    pipeline_export_import(root_dir)
    dedicated_models_export_import(root_dir)
    export_import_with_reshape(root_dir, args.prompt)


if "__main__" == __name__:
    main()
