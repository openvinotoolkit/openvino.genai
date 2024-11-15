
from optimum.intel import OVPipelineForText2Image, OVPipelineForImage2Image, OVPipelineForInpainting
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image

import torch

seed = 42
generator = torch.Generator(device='cpu').manual_seed(seed)

pipe = OVPipelineForInpainting.from_pretrained("/home/devuser/ilavreno/models/runwayml-stable-diffusion-inpainting")

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
image_url = "/home/devuser/ilavreno/models/initial_image.png"
mask_url = "/home/devuser/ilavreno/models/mask.png"
image = load_image(image_url)
mask_image = load_image(mask_url)
H, W = image.size[1], image.size[0]

images = pipe(prompt, image=image, mask_image=mask_image, width=W, height=H, generator=generator).images

images[0].save("/home/devuser/ilavreno/openvino.genai/optimum_image.bmp")
