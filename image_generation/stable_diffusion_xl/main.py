import numpy as np
from pathlib import Path
from openvino.runtime import Core
from optimum.intel.openvino import OVStableDiffusionXLPipeline

#Specify model dir
model_dir = Path("openvino-sd-xl-base-1.0")
#model_dir = Path("openvino-ssd-1b") #ssd-1b
#model_dir = Path("openvino-sdxl-turbo") #sdxl-turbo

#Define model parameters
core = Core()
device = "GPU"
text_prompt = "valley in the Alps at sunset, epic vista, beautiful landscape, 4k, 8k"
neg_prompt = None
#neg_prompt = "ugly, blurry, poor quality" #Negative prompt for SSD-B1 model
num_inference_steps = 25
seed = 42

#Define and load the model
text2image_pipe = OVStableDiffusionXLPipeline.from_pretrained(model_dir, device=device)

# Run txt2img inference pipeline
image = text2image_pipe(text_prompt, num_inference_steps=15, height=512, width=512, generator=np.random.RandomState(seed)).images[0]

#Run sdxl-turbo inference pipeline (guidance_scale set to 0)
#image = text2image_pipe(prompt, num_inference_steps=1, height=512, width=512, guidance_scale=0.0, generator=np.random.RandomState(987)).images[0]

image.save("result.png")