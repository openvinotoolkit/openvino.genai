#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai

import numpy as np

from openvino import Tensor
from pathlib import Path
from PIL import Image

def decrypt_model(model_dir, model_file_name, weights_file_name):
    with open(model_dir / model_file_name, "r") as file:
        model = file.read()
    # decrypt model

    with open(model_dir / weights_file_name, "rb") as file:
        binary_data = file.read()
    # decrypt weights
    weights = np.frombuffer(binary_data, dtype=np.uint8).astype(np.uint8)

    return model, Tensor(weights)

def read_tokenizer(model_dir):
    tokenizer_model_name = 'openvino_tokenizer.xml'
    tokenizer_weights_name = 'openvino_tokenizer.bin'
    tokenizer_model, tokenizer_weights = decrypt_model(model_dir, tokenizer_model_name, tokenizer_weights_name)

    detokenizer_model_name = 'openvino_detokenizer.xml'
    detokenizer_weights_name = 'openvino_detokenizer.bin'
    detokenizer_model, detokenizer_weights = decrypt_model(model_dir, detokenizer_model_name, detokenizer_weights_name)

    return openvino_genai.Tokenizer(tokenizer_model, tokenizer_weights, detokenizer_model, detokenizer_weights)


# here is example how to make cache de-encryption based on base64
import base64

def encrypt_base64(src: bytes):
    return base64.b64encode(src)

def decrypt_base64(src: bytes):
    return base64.b64decode(src)

def get_config_for_cache_encryption():
    config_cache = dict()
    config_cache["CACHE_DIR"] = "cache"
    config_cache["CACHE_ENCRYPTION_CALLBACKS"] = [encrypt_base64, decrypt_base64]
    config_cache["CACHE_MODE"] = "OPTIMIZE_SIZE"
    return config_cache

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')

    parser.add_argument('device', nargs='?', default='CPU')

    args = parser.parse_args()

    device = args.device
    model_dir = Path(args.model_dir)

    config = dict()
    if device == "GPU":
        # Cache compiled models on disk for GPU to save time on the
        # next run. It's not beneficial for CPU.
        config = get_config_for_cache_encryption()

    text_encoder_model, text_encoder_weights = decrypt_model(model_dir / "text_encoder", 'openvino_model.xml', 'openvino_model.bin')
    text_encoder_tokenizer = read_tokenizer(model_dir / "tokenizer")

    text_encoder_2_model, text_encoder_2_weights = decrypt_model(model_dir / "text_encoder_2", 'openvino_model.xml', 'openvino_model.bin')
    text_encoder_2_tokenizer = read_tokenizer(model_dir / "tokenizer_2")

    unet_model, unet_weights = decrypt_model(model_dir / "unet", 'openvino_model.xml', 'openvino_model.bin')
    vae_decoder_model, vae_decoder_weights = decrypt_model(model_dir / "vae_decoder", 'openvino_model.xml', 'openvino_model.bin')

    text_encoder = openvino_genai.CLIPTextModel(
        text_encoder_model,
        text_encoder_weights,
        openvino_genai.CLIPTextModel.Config(model_dir / "text_encoder" / "config.json"), 
        text_encoder_tokenizer, device, **config)
    text_encoder_2 = openvino_genai.CLIPTextModelWithProjection(
        text_encoder_2_model,
        text_encoder_2_weights,
        openvino_genai.CLIPTextModelWithProjection.Config(model_dir / "text_encoder_2" / "config.json"), 
        text_encoder_2_tokenizer, device, **config)
    
    vae = openvino_genai.AutoencoderKL(
        vae_decoder_model,
        vae_decoder_weights,
        openvino_genai.AutoencoderKL.Config(model_dir / "vae_decoder" / "config.json"),
        device, **config)

    unet = openvino_genai.UNet2DConditionModel(
        unet_model,
        unet_weights,
        openvino_genai.UNet2DConditionModel.Config(model_dir / "unet" / "config.json"),
        vae.get_vae_scale_factor(),
        device, **config)


    pipe = openvino_genai.Text2ImagePipeline.stable_diffusion_xl(
        scheduler=openvino_genai.Scheduler.from_config(model_dir / "scheduler" / "scheduler_config.json"),
        clip_text_model=text_encoder,
        clip_text_model_with_projection=text_encoder_2,
        unet=unet,
        vae=vae,
    )

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

if '__main__' == __name__:
    main()
