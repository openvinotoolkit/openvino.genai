from pathlib import Path
from openvino.runtime import Core
from diffusers.schedulers import LMSDiscreteScheduler
from transformers import CLIPTokenizer
from OVLatentConsistencyModelPipeline import OVLatentConsistencyModelPipeline
import torch
from diffusers import DiffusionPipeline

#Specify model dir
model_dir = Path("model/")

#Define model parameters
core = Core()
device = "GPU"
prompt = "a beautiful pink unicorn, 8k"
num_inference_steps = 4
torch.manual_seed(1234567)

#Compile the model
text_enc = core.compile_model(model_dir / "text_encoder.xml", device)
unet_model = core.compile_model(model_dir / "unet.xml", device)
vae_decoder = core.compile_model(model_dir / "vae_decoder.xml", device)

#Define LCM scheduler, tokenizer, and OpenVINO pipeline
#TBD - access pre-saved scheduler and tokenizer
pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
scheduler = pipe.scheduler
tokenizer = pipe.tokenizer

ov_pipe = OVLatentConsistencyModelPipeline(
    tokenizer=tokenizer,
    text_encoder=text_enc,
    unet=unet_model,
    vae_decoder=vae_decoder,
    scheduler=scheduler,
    feature_extractor=tokenizer, #feature_extractor,
    safety_checker=None
    #safety_checker=safety_checker,
)

print("Running inference")

# Run txt2img inference pipeline
images = ov_pipe(
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    guidance_scale=8.0,
    lcm_origin_steps=50,
    output_type="pil",
    height=512,
    width=512,
).images

final_image = images[0]
final_image.save('result.png')
final_image