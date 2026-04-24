---
name: vlm-model-enabler
description: "Enables VLM models for GenAI VLM pipelines."
argument-hint: "huggingface model_id and task (e.g. google/gemma-3n-E2B-it image-text-to-text). Alternatively directory to already converted OpenVINO IR model model_dir"
---

# VLM Model Enabler

## Input

Either:
- HuggingFace `model_id` and `task` (e.g. `google/gemma-3n-E2B-it image-text-to-text`)
- Path to an already exported OpenVINO IR model directory

## Working Mode

- Create working directory `.model_enabler/` for all intermediate assets, scripts, and investigation notes.
- Prefer calling tools over custom bash commands.
- Create Python files for experiments and testing — run them instead of inline bash/python snippets.
- Write investigation progress and findings to `.md` files in `.model_enabler/`.
- Use the who-what-benchmark (wwb) tool for final accuracy testing. The baseline is optimum-intel inference.

## Reference Documents

- [genai-vlm-architecture.md](genai-vlm-architecture.md) — GenAI VLM pipeline architecture, interfaces, and new-model checklist
- [model-analysis.md](model-analysis.md) — Instructions for model architecture analysis
- [model-inference-text-to-text.md](model-inference-text-to-text.md) — Text-only mode enablement
- [model-inference-image-text.md](model-inference-image-text.md) — Text-image mode enablement

## Enablement Steps

Execute steps in order. Depending on the model, steps can be skipped or reordered.

### Step 1 — Model Architecture Analysis

Read and follow [model-analysis.md](model-analysis.md).

Analyze the model architecture, export with optimum-intel, and produce an enablement design.
**Output:** `.model_enabler/<model_type>_architecture_analysis.md` — used as reference for all subsequent steps.

### Step 2 — Text-Only Inference

Read and follow [model-inference-text-to-text.md](model-inference-text-to-text.md).

Implement the minimum GenAI interfaces for VLMPipeline to work with text-only input (no images/videos).
**Acceptance:** Output matches optimum-intel exactly with greedy decoding.

### Step 3 — Text-Image Inference

Read and follow [model-inference-image-text.md](model-inference-image-text.md).

Implement image preprocessing, vision encoding, and embedding merge.
**Acceptance:** Output is semantically close to optimum-intel. Exact match not expected due to C++/Python preprocessing differences.

### Step 4 — Accuracy Verification with who-what-benchmark

Final accuracy gate across multiple samples.

Install wwb if not already installed:

```bash
pip install tools/who_what_benchmark
```

Generate baseline and evaluate:

```bash
# Generate baseline from optimum-intel
wwb --base-model <model_dir> \
    --gt-data .model_enabler/gt.csv \
    --model-type visual-text \
    --num-samples 20

# Evaluate GenAI
wwb --target-model <model_dir> \
    --gt-data .model_enabler/gt.csv \
    --model-type visual-text \
    --genai \
    --output .model_enabler/wwb_results \
    --num-samples 20
```

**Acceptance:** Similarity score close to 1.0. Investigate significant deviations.
