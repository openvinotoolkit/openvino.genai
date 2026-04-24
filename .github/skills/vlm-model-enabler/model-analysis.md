# Model Analysis — Supplementary Reference

This file provides **supplementary code samples** for Step 1 of the VLM Model Enabler skill.
The main workflow is in [SKILL.md](SKILL.md). Do not follow this file as a standalone procedure.

## Code Samples

### Print transformers model architecture

```python
from transformers import AutoModel, AutoProcessor
import torch

model = AutoModel.from_pretrained("<model_id>", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("<model_id>", trust_remote_code=True)
print(model)

from PIL import Image
import requests
image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
image = Image.open(requests.get(image_url, stream=True).raw)
inputs = processor(text="Describe this image.", images=[image], return_tensors="pt")
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
```

### Optimum-Intel reference inference

```python
from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoProcessor

model = OVModelForVisualCausalLM.from_pretrained("<model_dir>")
processor = AutoProcessor.from_pretrained("<model_dir>")
inputs = processor(text="What is this?", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(processor.decode(output[0], skip_special_tokens=True))
```
