# Image-Text Inference — Supplementary Reference

This file provides **supplementary details** for Step 3 of the VLM Model Enabler skill.
The main workflow is in [SKILL.md](SKILL.md). Do not follow this file as a standalone procedure.

For the list of reusable C++ preprocessing utilities (`clip.hpp` / `clip.cpp`) and the note on resize differences vs `transformers`, see [genai-vlm-architecture.md](genai-vlm-architecture.md) → *Preprocessing Utilities*.

## Test Script Templates

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
