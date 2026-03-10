---
name: genai-tools-fixer
description: "Fix known issues in OpenVINO GenAI tools (llm_bench, who-what-benchmark) that block new model support. Use after genai-model-checker reports a failure."
---

# GenAI Tools Fixer

Diagnoses and fixes known issues in the OpenVINO GenAI tools (`tools/llm_bench`, `tools/who_what_benchmark`) that prevent new models from working.

## When to Use

- After `genai-model-checker` reported a failure with a clear error trace
- Only when the failure originates from code under `tools/` (not from optimum-intel, openvino, or the model itself)

## Inputs

- **model_id**: HuggingFace model identifier
- **task**: optimum-cli export task
- **failure_step**: which step failed (`export`, `smoke_test`, `wwb`)
- **error_output**: the stderr/stdout from the failing step (full traceback)
- **model_ir_dir**: path to the exported model IR (e.g. `/tmp/genai-model-check/model_ir`)

## Scope

This skill fixes issues **only** in the GenAI tools directory:

- `tools/llm_bench/` — benchmark tool for inference smoke testing
- `tools/who_what_benchmark/` — accuracy evaluation tool

It does **not** fix:

- Export failures (optimum-intel / openvino issues)
- GenAI C++ library issues (`src/`)
- Model-inherent accuracy problems
- Test infrastructure (`tests/`)

## Diagnosis Procedure

### Step 1: Classify the failure

Read the error traceback and determine the origin:

| Error origin                                                                        | Action                               |
| ----------------------------------------------------------------------------------- | ------------------------------------ |
| Traceback points to files under `tools/llm_bench/`                                  | Proceed to Step 2                    |
| Traceback points to files under `tools/who_what_benchmark/`                         | Proceed to Step 2                    |
| Traceback points to `optimum-intel`, `openvino`, `transformers`, or other libraries | **Stop. Not fixable by this skill.** |
| Export step failed (before any tools ran)                                           | **Stop. Not fixable by this skill.** |

### Step 2: Identify the error pattern

Read the traceback carefully. Common patterns:

1. **"no use_case found"** — `get_use_case.py` cannot map the model's `config.json` `model_type` to a known use case
2. **"Unsupported model type"** — `model_loaders.py` in WWB cannot route the model type
3. **"Unsupported task"** — `wwb.py` cannot create an evaluator for the task type
4. **Other `KeyError`, `ValueError`, `RuntimeError`** with a clear message about missing model/type support

If the error does not match any recognizable pattern of missing model registration, **stop and report**. Do not attempt speculative fixes.

### Step 3: Gather model metadata

Read `<model_ir_dir>/config.json` and extract:

- `model_type` — the HuggingFace model type string
- `architectures` — the model architecture class names

Normalize the `model_type`: lowercase, replace `_` with `-`. This is how `llm_bench` processes it internally.

### Step 4: Determine the correct task mapping

Use this table to map the export task to the llm_bench task and WWB type:

| Export task                    | llm_bench task    | WWB model-type   |
| ------------------------------ | ----------------- | ---------------- |
| `text-generation-with-past`    | `text_gen`        | `text`           |
| `image-text-to-text`           | `visual_text_gen` | `visual-text`    |
| `text-to-image`                | `image_gen`       | `text-to-image`  |
| `image-to-image`               | `image_gen`       | `image-to-image` |
| `text-to-video`                | `video_gen`       | `text-to-video`  |
| `feature-extraction`           | `text_embed`      | `text-embedding` |
| `text-classification`          | `text_rerank`     | `text-reranking` |
| `automatic-speech-recognition` | `speech_to_text`  | N/A              |

### Step 5: Apply the appropriate fix

Read the relevant source files, understand the existing patterns, and apply the minimal change. See the **Fix Catalog** below for specific guidance per error.

## Fix Catalog

### Fix: Add model type to llm_bench USE_CASES registry

**Triggered by**: "no use_case found after checking all strategies" in `get_use_case.py`

**File**: `tools/llm_bench/llm_bench_utils/config_class.py`

**How it works**: The `USE_CASES` dictionary maps task names to lists of `UseCase*` objects. Each `UseCase*` is constructed with a list of model type strings. At runtime, `get_use_case_by_model_id()` checks if the normalized `model_type` from `config.json` starts with any registered string.

**Procedure**:

1. Open `tools/llm_bench/llm_bench_utils/config_class.py`
2. Find the `USE_CASES` dictionary
3. Locate the entry for the target task (e.g., `"text_gen"`)
4. Find the main `UseCase*` list for that task — the one with the longest model type list
5. Add the normalized `model_type` string in **alphabetical order** within the list
6. Verify: the normalized `model_type` must `startswith()` the string you added

**Special case — cross-task mapping**: If the model's export task implies one category (e.g., `text-generation-with-past` → `text_gen`) but the model actually needs a different pipeline (e.g., a VLM that exports as text-gen), add a mapping in `resolve_complex_model_types()` in `tools/llm_bench/llm_bench_utils/get_use_case.py` instead.

### Fix: Add model-type routing in WWB model_loaders

**Triggered by**: "Unsupported model type" in `model_loaders.py`

**File**: `tools/who_what_benchmark/whowhatbench/model_loaders.py`

**How it works**: The `load_model()` function dispatches based on the WWB `model_type` string (e.g., `"text"`, `"visual-text"`). Standard task types are already handled.

**This is typically NOT needed** for new models because the WWB model-type is the broad task category (not the HF model_type). A new text LLM uses `--model-type text` which is already supported. Only add routing if an entirely new task category is introduced.

### Fix: Add special-case model handling in llm_bench

**Triggered by**: Pipeline creation errors in `ov_utils.py` after use_case detection succeeded

**File**: `tools/llm_bench/llm_bench_utils/ov_utils.py` (or task-specific files in `tools/llm_bench/task/`)

**This requires understanding the specific error**. Read the traceback, identify which model creation function failed, and look at how similar models handle the same code path. If the fix is a simple conditional (e.g., processor selection for a VLM), apply it following the existing pattern. If the fix requires substantial new logic, **stop and report** — it needs manual engineering.

## Behavioral Rules

- **ONLY** modify files under `tools/llm_bench/` and `tools/who_what_benchmark/`
- Make **minimal** changes — add only what is needed for the specific model
- Follow existing code patterns and style in each file
- Do not refactor, clean up, or reorganize existing code
- Do not add comments explaining the fix unless adding a special-case branch (e.g., in `resolve_complex_model_types()`)
- If the error is ambiguous or the fix is unclear, **stop and report** rather than guessing
- If the error is not in the tools code path, **stop and report**

## Security Rules

- **NEVER** install any packages
- **NEVER** pass `--trust-remote-code` to any command
- **NEVER** execute model code or import model-specific modules
- **NEVER** modify model files or exported IR files
