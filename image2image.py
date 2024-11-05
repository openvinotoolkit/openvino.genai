
from optimum.intel import OVPipelineForText2Image, OVPipelineForImage2Image
from diffusers.utils import load_image

import torch

seed = 42
generator = torch.Generator(device='cpu').manual_seed(seed)

pipe = OVPipelineForImage2Image.from_pretrained(
    "/home/devuser/ilavreno/models/SimianLuo-LCM_Dreamshaper_v7"
)

prompt = "professional photo portrait of woman, highly detailed, hyper realistic, cinematic effects, soft lighting"
default_image_url = "/home/devuser/ilavreno/models/Twitter.png"
image = load_image(default_image_url)
H, W = image.size[1], image.size[0]

images = pipe(prompt, image, width=W, height=H, strength=0.6, generator=generator).images

images[0].save("/home/devuser/ilavreno/openvino.genai/optimum_image.bmp")
