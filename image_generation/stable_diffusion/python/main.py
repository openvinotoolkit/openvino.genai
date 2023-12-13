from pathlib import Path
from openvino.runtime import Core
from diffusers.schedulers import LMSDiscreteScheduler
from transformers import CLIPTokenizer
from OVStableDiffusionPipeline import OVStableDiffusionPipeline

#Specify model dir - SD 2.1, SD 1.5, or TinySD
model_dir = Path("tiny-sd")

#Define model parameters
core = Core()
device = "GPU"
text_prompt = "valley in the Alps at sunset, epic vista, beautiful landscape, 4k, 8k", 
negative_prompt = "low quality, ugly, deformed, blur"
num_steps = 25
seed = 42

#Compile the model
text_enc = core.compile_model(model_dir / "text_encoder.xml", device)
unet_model = core.compile_model(model_dir / "unet.xml", device)
vae_encoder = core.compile_model(model_dir / "vae_encoder.xml", device)
vae_decoder = core.compile_model(model_dir / "vae_decoder.xml", device)

#Define scheduler, tokenizer, and Stable Diffusion Pipeline
scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
ov_pipe = OVStableDiffusionPipeline(
    tokenizer=tokenizer,
    text_encoder=text_enc,
    unet=unet_model,
    vae_encoder=vae_encoder,
    vae_decoder=vae_decoder,
    scheduler=scheduler
)

# Run txt2img inference pipeline
result = ov_pipe(text_prompt, negative_prompt=negative_prompt, num_inference_steps=num_steps, 
                 seed=seed)

final_image = result['sample'][0]
final_image.save('result.png')
final_image