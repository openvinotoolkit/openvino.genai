import gc
from pathlib import Path

import torch
import openvino as ov
from diffusers import UNet3DConditionModel, AutoencoderKL, TextToVideoSDPipeline
from transformers import CLIPTextModel


def convert_text2video():
    output_dir = Path("text2video-1.7b")
    output_dir.mkdir(exist_ok=True)
    model_id = "ali-vilab/text-to-video-ms-1.7b"

    # 1. Text Encoder
    print("Converting Text Encoder...")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16"
    )
    text_encoder.eval()
    ov_text = ov.convert_model(
        text_encoder, example_input=torch.ones((1, 77), dtype=torch.long)
    )
    ov.save_model(ov_text, output_dir / "openvino_text_encoder_model.xml")
    del ov_text, text_encoder
    gc.collect()

    # 2. U-Net
    print("Converting U-Net...")
    unet = UNet3DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16, variant="fp16"
    )
    unet.eval()
    dummy_inputs = (
        torch.randn(1, 4, 16, 32, 32, dtype=torch.float16),
        torch.tensor(1, dtype=torch.int64),
        torch.randn(1, 77, 1024, dtype=torch.float16),
    )
    with torch.no_grad():
        ov_unet = ov.convert_model(unet, example_input=dummy_inputs)
    ov.save_model(ov_unet, output_dir / "openvino_model.xml")
    del ov_unet, unet, dummy_inputs
    gc.collect()

    # 3. VAE & Configs
    print("Converting VAE...")
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16, variant="fp16"
    )
    vae.eval()
    ov_vae = ov.convert_model(
        vae.decoder, example_input=torch.randn(1, 4, 32, 32, dtype=torch.float16)
    )
    ov.save_model(ov_vae, output_dir / "openvino_vae_decoder_model.xml")
    del ov_vae, vae
    gc.collect()

    # Save Scheduler and Tokenizer
    pipe = TextToVideoSDPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.scheduler.save_config(output_dir / "scheduler")
    pipe.tokenizer.save_pretrained(output_dir / "tokenizer")
    
    with open(output_dir / "model_index.json", "w") as f:
        f.write('{"_class_name": "TextToVideoSDPipeline", "_diffusers_version": "0.19.0"}')

    print("Conversion Complete.")


if __name__ == "__main__":
    convert_text2video()