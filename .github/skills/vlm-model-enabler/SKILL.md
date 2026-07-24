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

All intermediate assets for one model go in
`.model_enabler/<model_type>/`. Determine `<model_type>` from `config.json` and
create the directory if it does not exist. Never share this directory between
different model types; this keeps parallel and resumed enablement runs isolated.

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

For normal local development, install the current checkout with nightly
OpenVINO packages:

```bash
pip install --pre -U . --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
```

This is the default validation path. Do not validate edited source through a
previously installed OpenVINO GenAI wheel.

After building, verify:

```bash
python -c "import openvino, openvino_genai; print(openvino.__version__); print(openvino_genai.__file__)"
```

Confirm the imported module comes from this checkout's build/install output.

Use an OpenVINO source build only when the change requires it or a native
loading/ABI mismatch cannot be resolved with the installed runtime. In that
case, build OpenVINO and OpenVINO GenAI from source in the same environment as
described in `src/docs/BUILD.md`; do not treat adding an arbitrary library
directory to `LD_LIBRARY_PATH` as proof of compatibility.

Fix all compilation errors.

### 2.3 Verify

Create and run `.model_enabler/<model_type>/test_text_only_compare.py` — compare GenAI vs optimum-intel with `do_sample=False` on 3 prompts.

### Checkpoint

- [ ] Build succeeds
- [ ] `.model_enabler/<model_type>/test_text_only_compare.py` shows exact match on all prompts

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

Before reporting a tokenizer/export blocker, compare Hugging Face and OpenVINO
tokenization for every required image/chat special token. If the OpenVINO
tokenizer drops or splits added tokens, recover only the configured special
token IDs in the model-specific GenAI input embedder. This is needed because
`get_inputs_embeds()` uses image-placeholder IDs to locate where vision
embeddings must be merged:

1. Read `tokenizer.json` added tokens and the relevant token IDs from model and
   processor configuration.
2. Split the normalized prompt around required added-token strings.
3. Tokenize ordinary text spans with the OpenVINO tokenizer and splice the
   exact added-token IDs back into `input_ids`.
4. Verify image-placeholder positions exist and their count matches the vision
   embedding rows before merging.
5. Validate the recovery with real image-text generation and the Optimum
   comparison.

Do not hard-code a Hub repository ID or silently replace missing tokens.
For models exposing `image_pad_token_id`, assert or explicitly check that the
final tokenized `input_ids` contains that ID and that its occurrence count
equals the number of image-embedding rows inserted by the model-specific
embedder. A prompt containing only the textual placeholder is not sufficient.

### 3.3 Build and verify

Rebuild, then create and run:

- `.model_enabler/<model_type>/test_image_text_compare.py` — compare GenAI vs optimum-intel on 3 image prompts

If output differs, locate the first divergent component rather than judging
only decoded text. Compare, in order:

1. chat template and token IDs;
2. processor outputs, shapes, value ranges, layouts, masks, and spatial grids;
3. vision embeddings and projector/merger outputs;
4. image-token insertion positions and counts;
5. language-model inputs, logits, and generated token IDs.

Also inspect effective precision at every stage. Models may use `dtype`,
`torch_dtype`, nested vision/text precision fields, or cast inputs inside
`forward()`. Verify actual tensor and parameter dtypes; do not assume an fp32
export merely because the command requested fp32.

### Checkpoint

- [ ] Build succeeds
- [ ] `.model_enabler/<model_type>/test_image_text_compare.py` shows semantically similar outputs

---

## Step 4 — Repository Test Coverage

Every newly enabled model must add repository tests:

1. **Find the tiny-random model id**:
   - First infer it from the optimum-intel model description, tests, or release notes when available.
   - If it is not documented there, look it up directly on HuggingFace Hub.
   - Matching the `optimum-intel-internal-testing/tiny-random-*` prefix, for example `optimum-intel-internal-testing/tiny-random-gemma4-unified-it`.
2. **Add the model to VLM Python tests**:
   - Prefer extending `tests/python_tests/test_vlm_pipeline.py` with the tiny-random model id, prompt image tag, video tag if applicable, resolution, and any targeted skip/xfail entry required by an already-tracked issue.
   - Add a dedicated `tests/python_tests/test_<model_type>_*.py` only when the new model requires behavior that does not fit the shared VLM pipeline suite.
   - Use the existing VLM fixtures and converted-model cache helpers; do not use the full-size model in repository tests.
3. **Validate locally**:
   - Run the narrow pytest target for the added or modified tests from the activated virtual environment.
   - If the test cannot run locally, document the exact command, blocker, and expected CI coverage.

Apply model-specific dependency entries consistently to the relevant Linux,
manylinux, and Windows VLM workflows. Do not update a shared dependency for all
models when only the newly enabled architecture requires it.

Before adding a tiny-model ID to a test matrix, verify that it exists, is
accessible without a developer's private cache, preserves the real
architecture identity, and can execute the requested generation path. Never
add a guessed or not-yet-published Hub ID and then treat HTTP 401/403/404 as a
passing test. If repository policy requires a hosted fixture and publication
is unavailable, report a test-fixture blocker and leave enablement incomplete.
If the test infrastructure supports a deterministic local constructor, prefer
that over a newly uploaded model and cache it using repository conventions.

Do not report successful enablement after GenAI source changes unless the test
is added and its exact pytest result is recorded. If local execution is
blocked, add the test and report the exact blocker and expected CI coverage.
Confirm that the narrow pytest command collected at least one test; a passing
command with every test deselected is not validation.

---

## Final Deliverables

Before declaring the model enabled:

- [ ] `.model_analysis/<model_type>_analysis.md` exists with the `## GenAI Enablement Design` section
- [ ] All test scripts exist and pass in `.model_enabler/<model_type>/`
- [ ] Tiny-random repository coverage is added under `tests/python_tests/`
- [ ] Narrow pytest command/result, or the exact local blocker, is recorded
- [ ] `git diff --name-only` contains the intended source, test, and docs files only
- [ ] No debug prints, scratch artifacts, local absolute paths, or unrelated edits remain
