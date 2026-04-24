---
name: vlm-model-enabler
description: "Enables VLM models for GenAI VLM pipelines."
argument-hint: "huggingface model_id and task (e.g. google/gemma-3n-E2B-it image-text-to-text). Alternatively directory to already converted OpenVINO IR model model_dir"
---

# VLM Model Enabler

Enables a new VLM model in the GenAI VLM pipeline. Follows a strict 4-step workflow with checkpoints.

## Input

Either:
- HuggingFace `model_id` and `task` (e.g. `google/gemma-3n-E2B-it image-text-to-text`)
- Path to an already exported OpenVINO IR model directory

## Working Directory

All intermediate assets go in `.model_enabler/`. Create it if it does not exist.

## Rules

- Create `.py` files for all experiments — never use inline bash/python snippets.
- After each step, verify **all checkpoint files** exist before proceeding. If any are missing, create them.
- Use the todo list tool to track step progress.

## Reference

- [genai-vlm-architecture.md](genai-vlm-architecture.md) — GenAI VLM pipeline interfaces and new-model checklist. Read the "Adding a New Model — Checklist" section before Step 2.
- [model-analysis.md](model-analysis.md) — Supplementary code samples for model analysis (Step 1).
- [model-inference-text-to-text.md](model-inference-text-to-text.md) — Supplementary test templates and debugging tips for text-only mode (Step 2).
- [model-inference-image-text.md](model-inference-image-text.md) — Supplementary test templates, preprocessing utilities list, and debugging tips for image-text mode (Step 3).

---

## Step 1 — Model Analysis

**Goal:** Understand the model architecture and produce an enablement design.

### 1.1 Export (skip if model_dir provided)

```bash
optimum-cli export openvino --model <model_id> --task <task> .model_enabler/model_ir
```

### 1.2 Inspect exported IR models

Create and run `.model_enabler/inspect_ir.py` to print all inputs/outputs of every `openvino_*.xml`:

```python
from openvino import Core
from pathlib import Path
core = Core()
for xml in sorted(Path("<model_dir>").glob("openvino_*.xml")):
    m = core.read_model(xml)
    print(f"\n=== {xml.name} ===")
    for i in m.inputs:  print(f"  IN  {i.any_name}: {i.partial_shape} {i.element_type}")
    for o in m.outputs: print(f"  OUT {o.any_name}: {o.partial_shape} {o.element_type}")
```

### 1.3 Analyze transformers source

Locate `transformers/models/<model_type>/` and identify:
- Forward pass signature for each sub-model
- Image preprocessing (resize method, normalization constants, tiling)
- Special tokens for image/video placeholders
- Position ID generation

### 1.4 Analyze optimum-intel inference

Locate `optimum/intel/openvino/modeling_visual_language.py` and identify how the model class maps sub-models to inference requests.

### 1.5 Determine closest GenAI model

Compare with existing VLM implementations in `src/cpp/src/visual_language/`. Pick the closest one to use as reference.

### Checkpoint

Write `.model_enabler/<model_type>_analysis.md` with this structure:

```
## Model: <model_id> (<model_type>)
## Exported IR Models
<table of file, purpose, inputs, outputs>
## Preprocessing: <resize method>, normalize mean=<>, std=<>
## Special Tokens: image_token=<>, boi=<>, eoi=<>
## Closest GenAI Model: <name> — because <reason>
## Required Changes: <list of files to create/modify>
## Gaps: <anything not covered by existing infrastructure>
```

**Do not proceed to Step 2 until this file exists.**

---

## Step 2 — Text-Only Inference

**Goal:** VLMPipeline generates correct text with text-only input, matching optimum-intel exactly.

### 2.1 Implement C++ changes

Follow the checklist in [genai-vlm-architecture.md](genai-vlm-architecture.md) "Adding a New Model":
1. Add enum to `vlm_config.hpp`, string mapping to `vlm_config.cpp`
2. Create `<model_type>/classes.hpp` and `<model_type>/classes.cpp`
3. Register in `vision_encoder.cpp` and `inputs_embedder.cpp` factories
4. Add `friend class` to `inputs_embedder.hpp`
5. Implement stubs: `VisionEncoder::encode()` throws, `get_inputs_embeds()` handles text-only path
6. Handle any model-specific requirements (custom position IDs, extra LM inputs, etc.)

### 2.2 Build

```bash
pip install --pre -U . --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
```

Fix all compilation errors.

### 2.3 Verify

Create and run `.model_enabler/test_text_only.py` — basic smoke test.
Create and run `.model_enabler/test_text_only_compare.py` — compare GenAI vs optimum-intel with `do_sample=False` on 3 prompts.

### Checkpoint

- [ ] Build succeeds
- [ ] `.model_enabler/test_text_only.py` runs without errors
- [ ] `.model_enabler/test_text_only_compare.py` shows exact match on all prompts

**Do not proceed to Step 3 until text-only output matches optimum-intel exactly.**

---

## Step 3 — Image-Text Inference

**Goal:** VLMPipeline generates correct text with image input, semantically close to optimum-intel.

### 3.1 Implement vision preprocessing + encoding

Replace the VisionEncoder stub. Key references:
- `preprocessor_config.json` for resize/normalization params
- `clip.hpp`/`clip.cpp` for available resize utilities (`bilinear_resize`, `bicubic_resize`, etc.)
- The closest model implementation identified in Step 1

### 3.2 Implement embedding merge

Update `get_inputs_embeds()` to handle the non-empty images case: insert vision embeddings at placeholder token positions.

### 3.3 Build and verify

Rebuild, then create and run:
- `.model_enabler/test_image_text.py` — smoke test with a real image (use URL from openvino_notebooks)
- `.model_enabler/test_image_text_compare.py` — compare GenAI vs optimum-intel on 3 image prompts

### Checkpoint

- [ ] Build succeeds
- [ ] `.model_enabler/test_image_text.py` produces coherent image description
- [ ] `.model_enabler/test_image_text_compare.py` shows semantically similar outputs

---

## Step 4 — Accuracy Verification

**Goal:** Validate accuracy across multiple samples using who-what-benchmark.

```bash
pip install tools/who_what_benchmark

# HF baseline
wwb --base-model <model_id_or_dir> --gt-data .model_enabler/wwb/gt.csv \
    --model-type visual-text --num-samples 20 --hf

# Optimum-intel evaluation
wwb --target-model <model_dir> --gt-data .model_enabler/wwb/gt.csv \
    --model-type visual-text --num-samples 20 --output .model_enabler/wwb/optimum

# GenAI evaluation
wwb --target-model <model_dir> --gt-data .model_enabler/wwb/gt.csv \
    --model-type visual-text --genai --num-samples 20 --output .model_enabler/wwb/genai
```

### Checkpoint

- [ ] Optimum similarity ≥ 0.95 (if below, this is a model/export issue, not GenAI)
- [ ] GenAI similarity ≥ optimum similarity (GenAI should not be worse than optimum)
- [ ] `.model_enabler/wwb/` contains `gt.csv`, `optimum/metrics.csv`, `genai/metrics.csv`

---

## Final Deliverables

Before declaring the model enabled:
- [ ] `.model_enabler/<model_type>_analysis.md` exists
- [ ] All 4 test scripts exist and pass in `.model_enabler/`
- [ ] WWB results in `.model_enabler/wwb/`
- [ ] Docs updated: `site/docs/supported-models/_components/vlm-models-table/models.ts`
- [ ] `tools/who_what_benchmark/whowhatbench/model_loaders.py` updated if needed
