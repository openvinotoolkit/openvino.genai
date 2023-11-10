# Downloading OpenVINO IRs

The following models are compatible with OVStableDiffusionPipeline and main.py:
    - Stable Diffusion v1.5. Model ID: "prompthero/openjourney"
    - Stable Diffusion v2.1. Model ID: "stabilityai/stable-diffusion-2-1-base"
    - Tiny SD. Model ID: "segmind/tiny-sd"


Prior to running inference on the model, the OpenVINO IRs formats need to be downloaded, substitute the *model_id* and the *model_dir* in the below code snippet to download the above models:

```
import gc
from diffusers import StableDiffusionPipeline

model_id = "TBD"

#Extract models
pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cpu")
text_encoder = pipe.text_encoder
text_encoder.eval()
unet = pipe.unet
unet.eval()
vae = pipe.vae
vae.eval()

del pipe
gc.collect()

#Convert models

# Define a dir to save text-to-image models
txt2img_model_dir = Path("TBD")
txt2img_model_dir.mkdir(exist_ok=True)

from implementation.conversion_helper_utils import convert_encoder, convert_unet, convert_vae_decoder, convert_vae_encoder 

# Convert the Text-to-Image models from PyTorch -> Onnx -> OpenVINO
# 1. Convert the Text Encoder
txt_encoder_ov_path = txt2img_model_dir / "text_encoder.xml"
convert_encoder(text_encoder, txt_encoder_ov_path)
# 2. Convert the U-NET
unet_ov_path = txt2img_model_dir / "unet.xml"
convert_unet(unet, unet_ov_path, num_channels=4, width=96, height=96)
# 3. Convert the VAE encoder
vae_encoder_ov_path = txt2img_model_dir / "vae_encoder.xml"
convert_vae_encoder(vae, vae_encoder_ov_path, width=768, height=768)
# 4. Convert the VAE decoder
vae_decoder_ov_path = txt2img_model_dir / "vae_decoder.xml"
convert_vae_decoder(vae, vae_decoder_ov_path, width=96, height=96)
```

Run *main.py* to execute the inference pipeline of the model.