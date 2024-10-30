
from optimum.intel import OVPipelineForText2Image
from diffusers.utils import load_image

import torch
import numpy as np
import random

# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)
seed = 42
generator = torch.Generator(device='cpu').manual_seed(seed)
# t = torch.randn([2, 2], generator=generator, dtype=torch.float32)
# print(f'random tensor: {t}')

# generator = torch.Generator(device='cpu').manual_seed(seed)
# t = torch.randn([1, 4, 2, 2], generator=generator, dtype=torch.float32)
# print(f'random tensor: {t}')

pipe = OVPipelineForText2Image.from_pretrained(
    "/home/devuser/ilavreno/models/stabilityai-stable-diffusion-xl-base-1.0"
)

prompt = "professional photo portrait of woman, highly detailed, hyper realistic, cinematic effects, soft lighting"
# default_image_url = "/home/devuser/ilavreno/models/Twitter.png"
# image = load_image(default_image_url)

images = pipe(prompt, generator=generator).images

images[0].save("/home/devuser/ilavreno/openvino.genai/hf_image.bmp")