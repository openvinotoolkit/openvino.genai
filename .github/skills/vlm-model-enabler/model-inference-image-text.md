# Image-Text Inference — Supplementary Reference

This file provides **supplementary details** for Step 3 of the VLM Model Enabler skill.
The main workflow is in [SKILL.md](SKILL.md). Do not follow this file as a standalone procedure.

## Image Resize Differences

GenAI implements Pillow-style bicubic and bilinear resize in C++ (`clip.cpp`). Minor pixel-level differences are expected between GenAI and transformers preprocessing. Exact token-level output match is **not required** for image-text mode.

## Available Preprocessing Utilities (`clip.hpp` / `clip.cpp`)

- `bicubic_resize()` / `bilinear_resize()` — Pillow-style image resizing
- `center_crop()` — center crop to target dimensions
- `resize_and_pad_image()` — resize with center padding
- `get_image_patches()` — extract grid patches for multi-resolution
- `select_best_resolution()` — optimal resolution from candidates
- `clip_image_preprocess()` — normalize + convert to CHW
- `normalize_and_convert_to_chw()` — double-precision normalization
- `smart_resize()` — dynamic resolution with min/max pixel bounds (Qwen2VL-style). Implemented separately as `qwen2_vl_utils::smart_resize`.

## Test Script Templates

### Smoke test — `test_image_text.py`

```python
import numpy as np
import openvino_genai
import requests
from PIL import Image
from openvino import Tensor

model_dir = "<model_dir>"
image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"

image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
image_tensor = Tensor(np.array(image))

pipe = openvino_genai.VLMPipeline(model_dir, "CPU")
config = openvino_genai.GenerationConfig()
config.max_new_tokens = 100
config.do_sample = False

history = openvino_genai.ChatHistory()
history.append({"role": "user", "content": "Describe this image in detail."})
result = pipe.generate(history, images=[image_tensor], generation_config=config)
print("GenAI:", result.texts[0])
```

### Comparison — `test_image_text_compare.py`

```python
import numpy as np
import openvino_genai
import requests
from PIL import Image
from openvino import Tensor
from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoProcessor

model_dir = "<model_dir>"
max_new_tokens = 100
image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"

image_pil = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
image_tensor = Tensor(np.array(image_pil))
prompts = ["Describe this image in detail.", "What objects are visible?", "What colors dominate?"]

ov_model = OVModelForVisualCausalLM.from_pretrained(model_dir)
ov_processor = AutoProcessor.from_pretrained(model_dir)
pipe = openvino_genai.VLMPipeline(model_dir, "CPU")
config = openvino_genai.GenerationConfig()
config.max_new_tokens = max_new_tokens
config.do_sample = False

for prompt in prompts:
    inputs = ov_processor(text=prompt, images=[image_pil], return_tensors="pt")
    output = ov_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    oi_text = ov_processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    history = openvino_genai.ChatHistory()
    history.append({"role": "user", "content": prompt})
    result = pipe.generate(history, images=[image_tensor], generation_config=config)
    genai_text = result.texts[0]

    oi_words = set(oi_text.lower().split())
    genai_words = set(genai_text.lower().split())
    overlap = len(oi_words & genai_words) / max(len(oi_words), len(genai_words)) if oi_words else 0

    print(f"\nPrompt: {prompt}")
    print(f"  OI:    {oi_text[:200]}")
    print(f"  GenAI: {genai_text[:200]}")
    print(f"  Word overlap: {overlap:.0%}")
```

## Debugging Wrong Image Output

If output is unrelated to the image:
1. **Preprocessing** — compare pixel_values shape and value range against transformers reference
2. **Embedding merge** — verify image embeddings are inserted at correct positions
3. **num_image_tokens** — must match vision encoder output size
4. **Special tokens** — verify image boundary tokens match `config.json`
