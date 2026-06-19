---
name: vlm-model-enabler
description: "Enables VLM models for GenAI VLM pipelines."
argument-hint: "HuggingFace model_id and task (e.g. google/gemma-3n-E2B-it image-text-to-text), or a path to an already converted OpenVINO IR model directory (model_dir)."
---

# VLM Model Enabler

Enables a new VLM model in the GenAI VLM pipeline. Follows a strict 4-step workflow with checkpoints.

## Input

- HuggingFace `model_id` and `task` (e.g. `google/gemma-3n-E2B-it image-text-to-text`), or path to an already exported OpenVINO IR directory.
- **Recommended prerequisite**: `.model_analysis/<model_type>_analysis.md` produced by the `model-analysis` agent. If absent, the caller (or the user, if running this skill standalone) should invoke `model-analysis` first to avoid duplicating upstream inspection here. This skill does not invoke other agents itself.

## Working Directory

All intermediate assets go in `.model_enabler/`. Create it if it does not exist.

## Rules

- Create `.py` files for all experiments — avoid inline multi-line Python snippets; keep shell commands minimal and only for build/run steps.
- After each step, verify **all checkpoint files** exist before proceeding. If any are missing, create them.
- Use the todo list tool to track step progress.

## Reference

- [genai-vlm-architecture.md](genai-vlm-architecture.md) — GenAI VLM pipeline interfaces and new-model checklist. Read the "Adding a New Model — Checklist" section before Step 2.
- [model-inference-text-to-text.md](model-inference-text-to-text.md) — Supplementary test templates and debugging tips for text-only mode (Step 2).
- [model-inference-image-text.md](model-inference-image-text.md) — Supplementary test templates, preprocessing utilities list, and debugging tips for image-text mode (Step 3).

---

## Step 1 — GenAI Enablement Design

**Goal:** Map the model into the GenAI VLM pipeline using the upstream analysis report.

### 1.1 Obtain the analysis report

Read `.model_analysis/<model_type>_analysis.md`.

If it does not exist, prefer asking the caller to invoke the `model-analysis` agent first (it is the canonical source for upstream facts). If that is not possible — e.g. running this skill standalone without the agent available — produce an equivalent report yourself by following the procedure in [`.github/agents/model-analysis.agent.md`](../../agents/model-analysis.agent.md) and write the result to the same path. Do not skip this artifact; later steps depend on it.

### 1.2 Design the mapping

Using the report, decide:

- **Closest GenAI model**: compare the report's sub-model layout, preprocessing, and special tokens against existing implementations in `src/cpp/src/visual_language/*/classes.hpp`. Pick the closest one as the reference.
- **Required changes**: list files to create or modify (enum in `vlm_config.hpp`, new `<model_type>/classes.{hpp,cpp}`, factory registrations, etc.). Use [genai-vlm-architecture.md](genai-vlm-architecture.md) "Adding a New Model — Checklist" as the structure.
- **Gaps**: anything in the report that existing GenAI infrastructure does not cover (custom position IDs, extra LM inputs, dynamic image tiling, etc.).

### Checkpoint

Append a `## GenAI Enablement Design` section to `.model_analysis/<model_type>_analysis.md`:

```
## GenAI Enablement Design
- Closest GenAI model: <name> — because <reason>
- Required changes:
  - <file>: <what changes>
- Gaps: <items needing new infrastructure>
```

**Do not proceed to Step 2 until this section exists.**

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

Create and run `.model_enabler/test_text_only_compare.py` — compare GenAI vs optimum-intel with `do_sample=False` on 3 prompts.

### Checkpoint

- [ ] Build succeeds
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

- `.model_enabler/test_image_text_compare.py` — compare GenAI vs optimum-intel on 3 image prompts

### Checkpoint

- [ ] Build succeeds
- [ ] `.model_enabler/test_image_text_compare.py` shows semantically similar outputs

---

## Final Deliverables

Before declaring the model enabled:

- [ ] `.model_analysis/<model_type>_analysis.md` exists with the `## GenAI Enablement Design` section
- [ ] All test scripts exist and pass in `.model_enabler/`
