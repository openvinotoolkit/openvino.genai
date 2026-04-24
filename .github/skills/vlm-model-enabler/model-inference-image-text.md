# Model Inference — Text-Image Mode (VLMPipeline)

Enable VLMPipeline to produce correct output with text + image input.
This step builds on the text-only mode — the model type is already registered, factories are wired, and text-only inference works.

Use the architecture analysis from `.model_enabler/<model_type>_architecture_analysis.md` as the primary design reference.
See [genai-vlm-architecture.md](genai-vlm-architecture.md) for pipeline architecture.

## Goal

VLMPipeline generates correct text output for text + image prompts. Output should closely match optimum-intel.
This step is self-contained — it produces a working image-text model with verified accuracy.

## Known Limitation — Image Resize Differences

GenAI implements Pillow-style bicubic and bilinear resize in C++ (`clip.cpp`). While the implementation mirrors Pillow's fixed-point precision arithmetic, **minor pixel-level differences are expected** between GenAI and transformers preprocessing. These come from:
- Floating-point rounding differences between C++ and Python
- Different interpolation paths (GenAI C++ vs. Pillow/torchvision)
- Some models use OV IR-based preprocessing (e.g., `ov::op::v11::Interpolate`) which may differ from Pillow

This means **exact token-level output match is not required** for the image-text step. Instead, verify that:
1. Preprocessed image tensors are numerically close (use tolerance-based comparison)
2. Generated text is semantically correct and similar to optimum-intel output

## Step 1 — Capture Transformers Preprocessing Reference

Before implementing C++ preprocessing, capture the exact transformers preprocessor output as a reference.

Create `.model_enabler/capture_preprocessing_reference.py`:

```python
import numpy as np
import torch
import requests
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor

model_dir = "<model_dir>"
image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"

processor = AutoProcessor.from_pretrained(model_dir)
image = Image.open(requests.get(image_url, stream=True).raw)

print(f"Original image size: {image.size} (W x H)")

# Run the processor to get preprocessed tensors
inputs = processor(text="Describe this image in detail.", images=[image], return_tensors="pt")

print("\n=== Preprocessor outputs ===")
for key, val in inputs.items():
    if isinstance(val, torch.Tensor):
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}, "
              f"min={val.min().item():.6f}, max={val.max().item():.6f}, "
              f"mean={val.mean().item():.6f}")
    elif isinstance(val, list):
        print(f"  {key}: list of {len(val)} items")
    else:
        print(f"  {key}: {type(val).__name__}")

# Save pixel_values for later comparison with GenAI preprocessing
if "pixel_values" in inputs:
    pixel_values = inputs["pixel_values"]
    if isinstance(pixel_values, torch.Tensor):
        np.save(".model_enabler/reference_pixel_values.npy", pixel_values.numpy())
        print(f"\nSaved reference pixel_values: {pixel_values.shape}")
    elif isinstance(pixel_values, list):
        for i, pv in enumerate(pixel_values):
            arr = pv.numpy() if isinstance(pv, torch.Tensor) else np.array(pv)
            np.save(f".model_enabler/reference_pixel_values_{i}.npy", arr)
            print(f"\nSaved reference pixel_values_{i}: {arr.shape}")

# Also inspect the processor class hierarchy for analysis
print(f"\n=== Processor type ===")
print(f"  Processor class: {type(processor).__name__}")
if hasattr(processor, 'image_processor'):
    ip = processor.image_processor
    print(f"  Image processor class: {type(ip).__name__}")
    for attr in ['size', 'crop_size', 'image_mean', 'image_std', 'rescale_factor',
                 'do_resize', 'do_center_crop', 'do_normalize', 'do_rescale',
                 'resample', 'patch_size', 'min_pixels', 'max_pixels']:
        if hasattr(ip, attr):
            print(f"    {attr}: {getattr(ip, attr)}")
```

Run this script and analyze the output. This gives:
- Exact preprocessor config values (resize target, normalization constants, crop settings)
- Reference `pixel_values` tensor to compare against GenAI output
- Image processor class name (to find the transformers source for analysis)

If the architecture analysis does not fully describe the preprocessing pipeline, analyze the transformers image processor source at:
```
<python_env>/lib/python3.*/site-packages/transformers/models/<model_type>/image_processing_*.py
```

## Step 2 — Implement Vision Preprocessing in GenAI

Based on the architecture analysis and the reference output from Step 1, implement the image preprocessing in the `VisionEncoder` subclass.

Key decisions based on the architecture analysis:
- What resize method does the model use? (bicubic → `bicubic_resize()`, bilinear → `bilinear_resize()`)
- Does the model use center crop, padding, or direct resize?
- What normalization constants are used? (`image_mean`, `image_std` from `ProcessorConfig`)
- Does the model use image tiling/patching? (e.g., LLaVA-Next grid, MiniCPM slicing, Qwen2VL dynamic resolution)
- Are there additional preprocessing steps? (rescaling, channel reordering, temporal patching for video)

Available GenAI preprocessing utilities in `clip.hpp` / `clip.cpp`:
- `bicubic_resize()` / `bilinear_resize()` — Pillow-style image resizing
- `center_crop()` — center crop to target dimensions
- `resize_and_pad_image()` — resize with center padding
- `get_image_patches()` — extract grid patches for multi-resolution
- `select_best_resolution()` — optimal resolution from candidates
- `clip_image_preprocess()` — normalize + convert to CHW
- `normalize_and_convert_to_chw()` — double-precision normalization (for sensitive models)
- `smart_resize()` — dynamic resolution with min/max pixel bounds (Qwen2VL-style)

Replace the VisionEncoder stub from the text-only step with a real preprocessing implementation.
At this stage, implement **only the preprocessing** — convert the input image to the normalized tensor format expected by the vision encoder IR model.

## Step 3 — Verify Preprocessing Against Reference

Create `.model_enabler/verify_preprocessing.py` to compare GenAI preprocessing output against the transformers reference:

```python
import numpy as np
import openvino_genai
import requests
from pathlib import Path
from PIL import Image
from openvino import Tensor as OVTensor

model_dir = "<model_dir>"
image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"

image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
image_data = np.array(image)
image_tensor = OVTensor(image_data)

# Load transformers reference
ref = np.load(".model_enabler/reference_pixel_values.npy")
print(f"Reference shape: {ref.shape}, min={ref.min():.6f}, max={ref.max():.6f}")

# GenAI preprocessing produces tensors inside VisionEncoder::encode()
# To compare, add temporary debug output or use the internal preprocessing
# function directly. The exact approach depends on model implementation.

# Compare shapes and values
# NOTE: Minor differences are expected due to resize implementation differences.
# Typical tolerance: atol=0.01 for normalized pixel values.
# If differences are larger, investigate the resize or normalization path.
print("\n=== Comparison ===")
print(f"Shape match: {ref.shape == genai_preprocessed.shape}")
if ref.shape == genai_preprocessed.shape:
    abs_diff = np.abs(ref - genai_preprocessed)
    print(f"Max absolute difference: {abs_diff.max():.6f}")
    print(f"Mean absolute difference: {abs_diff.mean():.6f}")
    close = np.allclose(ref, genai_preprocessed, atol=0.01)
    print(f"Close (atol=0.01): {close}")
    if not close:
        print("Expected: minor differences from C++ vs Python resize are normal.")
        print("Investigate if max difference > 0.05 or shapes don't match.")
```

Adapt this script to the specific model — the method for extracting GenAI's preprocessed tensor depends on the implementation.
Key validation criteria:
- **Shapes must match** — same spatial dimensions and channel count
- **Values should be close** — `atol=0.01` for normalized floats is typical
- **Larger differences (> 0.05 max abs diff)** indicate a bug in resize, normalization, or padding logic

## Step 4 — Implement Vision Encoding and Embedding Merge

After preprocessing is verified, complete the `VisionEncoder::encode()` implementation:
1. Run the preprocessed tensor through the vision encoder IR model
2. Populate `EncodedImage` fields (which fields depends on the model — see architecture analysis)
3. Set `num_image_tokens` for prompt expansion

Then update `IInputsEmbedder::get_inputs_embeds()` to handle the non-empty images case:
1. Replace the `OPENVINO_THROW` placeholder from the text-only step
2. Implement the model-specific merge logic: insert vision embeddings at image placeholder positions in the text embedding sequence
3. Refer to the closest existing model implementation identified in the architecture analysis

Key decisions based on the architecture analysis:
- How are image embeddings inserted into the text sequence? (replace placeholder tokens, concatenate, interleave)
- Does the model use additional projection/resampler models between vision encoder and LLM?
- Are there special separator embeddings between image patches? (e.g., `image_newline` for LLaVA-Next, `sub_GN`/`glb_GN` for Phi3)
- Does the model need custom position IDs for image tokens?

Build and fix compilation errors after implementation.

## Verification

### Smoke test with image

Create `.model_enabler/test_image_text.py`:

```python
import numpy as np
import openvino_genai
import requests
from PIL import Image
from openvino import Tensor

model_dir = "<model_dir>"
image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"

image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
image_data = np.array(image)
image_tensor = Tensor(image_data)

pipe = openvino_genai.VLMPipeline(model_dir, "CPU")

config = openvino_genai.GenerationConfig()
config.max_new_tokens = 100
config.do_sample = False

history = openvino_genai.ChatHistory()
history.append({"role": "user", "content": "Describe this image in detail."})

result = pipe.generate(history, images=[image_tensor], generation_config=config)
print("GenAI output:", result.texts[0])
```

### Compare with optimum-intel

Create `.model_enabler/test_image_text_compare.py`:

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
image_data = np.array(image_pil)
image_tensor = Tensor(image_data)

prompts = [
    "Describe this image in detail.",
    "What objects are visible in this image?",
    "What colors dominate the image?",
]

# --- Optimum-Intel ---
ov_model = OVModelForVisualCausalLM.from_pretrained(model_dir)
ov_processor = AutoProcessor.from_pretrained(model_dir)

# --- GenAI ---
pipe = openvino_genai.VLMPipeline(model_dir, "CPU")
config = openvino_genai.GenerationConfig()
config.max_new_tokens = max_new_tokens
config.do_sample = False

for prompt in prompts:
    # Optimum-Intel
    inputs = ov_processor(text=prompt, images=[image_pil], return_tensors="pt")
    output = ov_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    oi_text = ov_processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # GenAI
    history = openvino_genai.ChatHistory()
    history.append({"role": "user", "content": prompt})
    result = pipe.generate(history, images=[image_tensor], generation_config=config)
    genai_text = result.texts[0]

    # Due to preprocessing differences, exact match is not expected.
    # Compare semantically — outputs should describe the same content.
    print(f"\nPrompt: {prompt}")
    print(f"  OI:    {oi_text[:300]}")
    print(f"  GenAI: {genai_text[:300]}")

    # Simple heuristic: check if outputs share significant overlap
    oi_words = set(oi_text.lower().split())
    genai_words = set(genai_text.lower().split())
    if oi_words and genai_words:
        overlap = len(oi_words & genai_words) / max(len(oi_words), len(genai_words))
        print(f"  Word overlap: {overlap:.1%}")
```

For image-text mode, **exact match is not expected** due to preprocessing differences.
Instead verify:
1. GenAI produces coherent, relevant descriptions of the image
2. Output is semantically similar to optimum-intel (describes same objects/scene)
3. No crashes or errors during inference

If output is completely wrong or unrelated to the image, investigate:
1. **Preprocessing** — re-run Step 3 comparison, check shapes and value ranges
2. **Embedding merge** — verify image embeddings are inserted at correct positions
3. **num_image_tokens** — must match the actual vision encoder output size
4. **Special tokens** — verify image boundary tokens match `config.json`
