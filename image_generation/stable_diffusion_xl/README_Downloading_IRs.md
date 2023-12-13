# Downloading OpenVINO IRs

The following models are compatible with OVStableDiffusionPipeline and main.py:
    - Stable Diffusion XL
    - Segmind Stable Diffusion 1B (SSD-1B)
    - SDXL-turbo

Prior to running inference on the model, the OpenVINO IRs formats need to be downloaded, substitute the *model_id* and the *model_dir* in the below code snippet to download the above models. 

Run *main.py* to execute the inference pipeline of the model.

## OV SD XL Model
```
from pathlib import Path
from optimum.intel.openvino import OVStableDiffusionXLPipeline
import gc

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
model_dir = Path("openvino-sd-xl-base-1.0")
#model_id = "segmind/SSD-1B"
#model_dir = Path("openvino-ssd-1b")

if not model_dir.exists():
    text2image_pipe = OVStableDiffusionXLPipeline.from_pretrained(model_id, compile=False, device="GPU", export=True)
    text2image_pipe.half()
    text2image_pipe.save_pretrained(model_dir)
    text2image_pipe.compile()
```

## OV SSD 1B Model
```
from pathlib import Path
from optimum.intel.openvino import OVStableDiffusionXLPipeline

model_id = "segmind/SSD-1B"
model_dir = Path("openvino-ssd-1b")

if not model_dir.exists():
    text2image_pipe = OVStableDiffusionXLPipeline.from_pretrained(model_id, compile=False, device="GPU", export=True)
    text2image_pipe.half()
    text2image_pipe.save_pretrained(model_dir)
    text2image_pipe.compile()
    gc.collect()
```

## SDXL Turbo