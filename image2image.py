
from optimum.intel import OVPipelineForImage2Image
from diffusers.utils import load_image

import torch
import numpy as np
import random
 
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
 
pipe = OVPipelineForImage2Image.from_pretrained(
    "/home/devuser/ilavreno/models/lcm_dreamshaper_v7/FP16"
)
 
prompt = "professional photo portrait of woman, highly detailed, hyper realistic, cinematic effects, soft lighting"
default_image_url = "/home/devuser/ilavreno/models/Twitter.png"
image = load_image(default_image_url)

images = pipe(prompt, image, strength=0.4, num_inference_steps=20).images

images[0].save("/home/devuser/ilavreno/openvino.genai/hf_image.png")
