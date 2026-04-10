---
name: model-checker
description: "Validate a newly supported optimum-intel model with OpenVINO GenAI. Use when: checking new model support, verifying model export to OpenVINO IR, running GenAI inference test with llm_bench, benchmarking model accuracy with who-what-benchmark."
argument-hint: "model_id and task (e.g. tencent/HY-MT1.5-1.8B text-generation-with-past)"
---

# Model Checker

Validates that a HuggingFace model exported via optimum-intel works correctly with OpenVINO GenAI pipelines and passes accuracy benchmarks.

## When to Use

- A new model was added to optimum-intel and needs GenAI validation
- Verify a HuggingFace model exports to OpenVINO IR and runs inference
- Check model accuracy after conversion using who-what-benchmark

## Inputs

The user must provide:

- **model_id**: HuggingFace model identifier (e.g. `tencent/HY-MT1.5-1.8B`)
- **task**: optimum-cli export task. Supported values:
  - `text-generation-with-past`
  - `image-text-to-text`
  - `text-to-image`
  - `image-to-image`
  - `feature-extraction`
  - `text-classification`
  - `text-to-video`
  - `automatic-speech-recognition`

## Prerequisites

Activate the Python virtual environment before running any commands.

1. **Locate the virtual environment** — check for common directories at the repository root: `.venv/`, `venv/`, `env/`. Use `list_dir` to find it. If none is found, ask the user for its location.
2. **Activate** based on the current platform:
   - **Linux/macOS**: `source <venv_path>/bin/activate`
   - **Windows (cmd)**: `<venv_path>\Scripts\activate.bat`
   - **Windows (PowerShell)**: `<venv_path>\Scripts\Activate.ps1`

## Procedure

### Step 1: Run check_model.py

Run the checker script from the repository root:

```
python3 .github/skills/model-checker/scripts/check_model.py \
    --model-id <model_id> \
    --task <export_task> \
    --work-dir .model_enabler/model_checker
```

Run `python3 .github/skills/model-checker/scripts/check_model.py --help` for the full argument reference including defaults. The `--work-dir` is where all intermediate files, logs, and outputs will be stored. Do not pipe with any additional logging or redirection — the script handles its own logging.

#### Skip flags (for re-runs after a fix)

When a previous run already passed some steps (e.g. export succeeded but inference test failed), use skip flags to avoid repeating expensive passed steps:

- `--skip-export` — reuse existing IR in `<work-dir>/model_ir` instead of re-exporting (avoids re-downloading weights)
- `--skip-llm-bench` — skip the llm_bench inference test
- `--skip-wwb` — skip the who-what-benchmark accuracy check

Do **not** use skip flags on the first run. Only use them when retrying after a targeted fix.

### Step 2: Interpret Results

The script logs progress for each step and exits with code 0 (pass) or non-zero (fail).

**Pass criteria:**

- Export: exit code 0
- Inference test (llm_bench): exit code 0, metrics line logged
- WWB accuracy (three sub-steps, all must pass):
  1. HF ground truth generation: exit code 0
  2. Optimum target evaluation: similarity ≥ `SIMILARITY_THRESHOLD`
  3. GenAI target evaluation: similarity ≥ `SIMILARITY_THRESHOLD`

  Note: the WWB step is skipped automatically for `automatic-speech-recognition` (no WWB support).

**Log files:** each tool writes its own dedicated log; paths are printed during execution. When a step fails, read the corresponding log for the full traceback and context before drawing any conclusions.

### Step 3: Report Results

Results format:

- **Model**: `<model_id>` (`<task>`)
- **Validation**: PASSED / FAILED
- **Performance** (if passed):
  - 1st token latency, 2nd token latency, throughput
  - Optimum similarity / GenAI similarity (if applicable)
- **Logs**: paths to export log, llm_bench log, WWB logs
- **Failed step analysis** (if failed): summary of the failure and relevant log path for details

### Security

- **NEVER** install any packages. Assume the environment is pre-configured.
- **NEVER** invoke `optimum-cli`, `wwb`, or `llm_bench` directly. Always go through `check_model.py`.
- **NEVER** modify `model_id` — pass it exactly as provided by the user.
