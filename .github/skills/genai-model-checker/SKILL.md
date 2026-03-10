---
name: genai-model-checker
description: "Validate a newly supported optimum-intel model with OpenVINO GenAI. Use when: checking new model support, verifying model export to OpenVINO IR, running GenAI inference smoke test, benchmarking model accuracy with who-what-benchmark."
argument-hint: "model_id and task (e.g. tencent/HY-MT1.5-1.8B text-generation-with-past)"
---

# GenAI Model Checker

Validates that a HuggingFace model exported via optimum-intel works correctly with OpenVINO GenAI pipelines and passes accuracy benchmarks.

## When to Use

- A new model was added to optimum-intel and needs GenAI validation
- Verify a HuggingFace model exports to OpenVINO IR and runs inference
- Check model accuracy after conversion using who-what-benchmark

## Inputs

The user must provide:

- **model_id**: HuggingFace model identifier (e.g. `tencent/HY-MT1.5-1.8B`)
- **task**: optimum-cli export task (e.g. `text-generation-with-past`, `image-text-to-text`)

## Procedure

### Step 0: Activate python virtual environment

```
source .venv/bin/activate
```

### Step 1: Run check_model.py

Run the checker script from the repository root:

```
python3 .github/skills/genai-model-checker/scripts/check_model.py \
    --model-id <model_id> \
    --task <export_task> \
    --work-dir /tmp/genai-model-check \
    --llm-bench-script tools/llm_bench/benchmark.py
```

Optional arguments:

- `--device CPU` (default: CPU)
- `--skip-wwb` to skip accuracy benchmarking and only run export + smoke test
- `--num-samples 32` to control WWB sample count

### Step 2: Interpret Results

The script prints a structured summary and exits with code 0 (pass) or 1 (fail).

**Pass criteria:**

- Export: exit code 0, IR files created
- Smoke test (llm_bench): exit code 0, output generated
- WWB accuracy: similarity score > 0.95 for text models, > 0.90 for image models

Report results to the user or a next agent. If a step failed, include the error output. Include full stack traces for debugging. Also include log files path for reference.

## Behavioral Rules

- **DO NOT** attempt to fix, patch, or work around failures. The sole purpose is to run the check and report results.
- **DO NOT** edit source code, configuration files, or model files to make a failing step pass.
- If a step fails, report the failure with the error output and stop. Do not retry with modified parameters or code changes.

## Execution Rules

- The script streams output live. Individual steps (export, WWB) can take **several minutes**.

## Security Rules

- **NEVER** install any packages. Assume the environment is pre-configured.
- **NEVER** pass `--trust-remote-code` to any command.
- **NEVER** construct shell commands via string concatenation. Always use the script.
- **NEVER** modify the model_id — pass it exactly as the user provided after validating format.
