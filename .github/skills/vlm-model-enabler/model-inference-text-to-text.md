# Model Inference — Text-Only Mode (VLMPipeline)

Enable VLMPipeline to produce correct text output with text-only input (no images/videos).
This is the entry point for new model enablement — get text generation working and matching optimum-intel before implementing vision.

Use the architecture analysis from `.model_enabler/<model_type>_architecture_analysis.md` as the primary design reference.
See [genai-vlm-architecture.md](genai-vlm-architecture.md) for pipeline architecture and the new-model checklist.

## Goal

VLMPipeline generates correct text output for text-only prompts. Output must match optimum-intel with greedy decoding.
This step is self-contained — it produces a working, testable model before the image enablement step begins.

## Implementation

Based on the architecture analysis, implement the minimum GenAI interfaces required for text-only inference.
Refer to [genai-vlm-architecture.md](genai-vlm-architecture.md) "Adding a New Model — Checklist" for the full list of registration and factory changes.

Key decisions to make based on the architecture analysis:
- Which existing model implementation is closest? Use it as reference.
- What `model_type` string does `config.json` use? Register the string-to-enum mapping accordingly.
- Does the model need custom `VLMConfig` fields (special tokens, embedding params)?
- Does the model need `token_type_ids`? If so, implement `get_inputs_embeds_with_token_type_ids()` and `has_token_type_ids()`.
- What is the image placeholder token? Set it in `normalize_prompt()` even though images are empty — the tag must be correct for the image step later.

For the text-only path:
- `VisionEncoder::encode()` — stub that throws (never called when images are empty)
- `encode_images()` — iterates over input images and calls the vision encoder; returns empty vector when no images are passed
- `normalize_prompt()` — returns prompt with empty image sequence when no images are present
- `get_inputs_embeds()` — when images are empty: tokenize prompt → get text embeddings via `EmbeddingsModel` → return directly (no merge logic needed)

Build the project after implementation and fix any compilation errors.

## Verification

### Quick smoke test

Create `.model_enabler/test_text_only.py`:

```python
import openvino_genai

model_dir = "<model_dir>"

pipe = openvino_genai.VLMPipeline(model_dir, "CPU")

config = openvino_genai.GenerationConfig()
config.max_new_tokens = 100
config.do_sample = False

history = openvino_genai.ChatHistory()
history.append({"role": "user", "content": "What is OpenVINO?"})

result = pipe.generate(history, generation_config=config)
print("GenAI output:", result.texts[0])
```

### Compare with optimum-intel

Create `.model_enabler/test_text_only_compare.py`:

```python
import openvino_genai
from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoProcessor

model_dir = "<model_dir>"
max_new_tokens = 100

prompts = [
    "What is OpenVINO?",
    "Explain the concept of attention mechanism in transformers.",
    "Write a short poem about AI.",
]

# --- Optimum-Intel ---
ov_model = OVModelForVisualCausalLM.from_pretrained(model_dir)
ov_processor = AutoProcessor.from_pretrained(model_dir)

# --- GenAI ---
pipe = openvino_genai.VLMPipeline(model_dir, "CPU")
config = openvino_genai.GenerationConfig()
config.max_new_tokens = max_new_tokens
config.do_sample = False

all_match = True
for prompt in prompts:
    # Optimum-Intel
    inputs = ov_processor(text=prompt, return_tensors="pt")
    output = ov_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    oi_text = ov_processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # GenAI
    history = openvino_genai.ChatHistory()
    history.append({"role": "user", "content": prompt})
    result = pipe.generate(history, generation_config=config)
    genai_text = result.texts[0]

    match = oi_text.strip() == genai_text.strip()
    all_match = all_match and match
    status = "MATCH" if match else "MISMATCH"
    print(f"\n[{status}] Prompt: {prompt[:50]}...")
    if not match:
        print(f"  OI:    {oi_text[:200]}")
        print(f"  GenAI: {genai_text[:200]}")

print(f"\nAll match: {all_match}")
```

Outputs must match exactly with greedy decoding (`do_sample=False`).
If outputs differ, investigate:
1. **Chat template** — compare tokenized input IDs between optimum-intel and GenAI
2. **Special tokens** — verify BOS/EOS token handling
3. **Embedding scale** — check `scale_emb` in VLMConfig
4. **Position IDs** — verify position ID generation matches
