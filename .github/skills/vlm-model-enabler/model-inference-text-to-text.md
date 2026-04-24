# Text-Only Inference — Supplementary Reference

This file provides **supplementary details** for Step 2 of the VLM Model Enabler skill.
The main workflow is in [SKILL.md](SKILL.md). Do not follow this file as a standalone procedure.

## Test Script Templates

### Smoke test — `test_text_only.py`

```python
import openvino_genai

pipe = openvino_genai.VLMPipeline("<model_dir>", "CPU")
config = openvino_genai.GenerationConfig()
config.max_new_tokens = 100
config.do_sample = False

history = openvino_genai.ChatHistory()
history.append({"role": "user", "content": "What is OpenVINO?"})
result = pipe.generate(history, generation_config=config)
print("GenAI:", result.texts[0])
```

### Comparison — `test_text_only_compare.py`

```python
import openvino_genai
from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoProcessor

model_dir = "<model_dir>"
max_new_tokens = 100
prompts = ["What is OpenVINO?", "Explain attention mechanisms.", "Write a short poem about AI."]

ov_model = OVModelForVisualCausalLM.from_pretrained(model_dir)
ov_processor = AutoProcessor.from_pretrained(model_dir)

pipe = openvino_genai.VLMPipeline(model_dir, "CPU")
config = openvino_genai.GenerationConfig()
config.max_new_tokens = max_new_tokens
config.do_sample = False

all_match = True
for prompt in prompts:
    inputs = ov_processor(text=prompt, return_tensors="pt")
    output = ov_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    oi_text = ov_processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    history = openvino_genai.ChatHistory()
    history.append({"role": "user", "content": prompt})
    result = pipe.generate(history, generation_config=config)
    genai_text = result.texts[0]

    match = oi_text.strip() == genai_text.strip()
    all_match = all_match and match
    print(f"[{'MATCH' if match else 'MISMATCH'}] {prompt[:50]}...")
    if not match:
        print(f"  OI:    {oi_text[:200]}")
        print(f"  GenAI: {genai_text[:200]}")

print(f"\nAll match: {all_match}")
```

## Debugging Mismatches

If text-only outputs differ from optimum-intel:
1. **Chat template** — compare tokenized input IDs between optimum-intel and GenAI
2. **Special tokens** — verify BOS/EOS token handling
3. **Embedding scale** — check `scale_emb` in VLMConfig
4. **Position IDs** — verify position ID generation matches
