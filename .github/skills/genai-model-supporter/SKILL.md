---
name: genai-model-supporter
description: "End-to-end orchestrator: validate a new model with OpenVINO GenAI, fix known tooling gaps, and re-validate. Combines genai-model-checker and genai-tools-fixer skills."
argument-hint: "model_id and task (e.g. tencent/HY-MT1.5-1.8B text-generation-with-past)"
---

# GenAI Model Supporter

Orchestrates end-to-end new model support: runs validation, applies fixes for known tooling gaps, and re-validates.

## When to Use

- Adding support for a new HuggingFace model in OpenVINO GenAI
- When you want automated check → fix → re-check cycle instead of manual iteration

## Inputs

The user must provide:

- **model_id**: HuggingFace model identifier (e.g. `tencent/HY-MT1.5-1.8B`)
- **task**: optimum-cli export task (e.g. `text-generation-with-past`, `image-text-to-text`)

## Procedure

### Phase 1: Initial Validation

Load and follow the `genai-model-checker` skill:

- Read `.github/skills/genai-model-checker/SKILL.md`
- Execute the full procedure with the provided `model_id` and `task`
- Collect the structured results

**If all steps pass**: Report success and stop. No fixes needed.

**If a step fails**: Proceed to Phase 2.

### Phase 2: Diagnose and Fix

Load the `genai-tools-fixer` skill:

- Read `.github/skills/genai-tools-fixer/SKILL.md`
- Pass the failure details: `model_id`, `task`, `failure_step`, `error_output`, and `model_ir_dir`
- Follow the fixer's diagnosis and fix procedures

**If the fixer determines the failure is not fixable** (error not in tools code, or ambiguous): Report the failure with full details and stop. No further action.

**If the fixer applies a fix**: Proceed to Phase 3.

### Phase 3: Re-Validation

Re-run the `genai-model-checker` procedure, but with `--skip-wwb` first to verify the fix works for the smoke test:

```
python3 .github/skills/genai-model-checker/scripts/check_model.py \
    --model-id <model_id> \
    --task <task> \
    --work-dir /tmp/genai-model-check \
    --llm-bench-script tools/llm_bench/benchmark.py \
    --skip-wwb
```

**If the smoke test still fails**: Revert the fix, report the failure, and stop. The issue requires manual investigation.

**If the smoke test passes**: Run the full check including WWB:

```
python3 .github/skills/genai-model-checker/scripts/check_model.py \
    --model-id <model_id> \
    --task <task> \
    --work-dir /tmp/genai-model-check \
    --llm-bench-script tools/llm_bench/benchmark.py
```

### Phase 4: Report

Provide a structured report:

```
## Model Support Report: <model_id>

**Task**: <task>
**Overall**: PASS / FAIL

### Steps
| Step | Result |
|------|--------|
| Export to OpenVINO IR | PASS/FAIL |
| Smoke test (llm_bench) | PASS/FAIL |
| WWB accuracy | PASS/FAIL/SKIPPED |

### Changes Made
- List of files modified with a brief description of each change
- Or "No changes needed" if the model worked out of the box

### Issues (if any)
- Description of any unfixable issues encountered
- Full error output for debugging
```

## Orchestration Rules

- **Maximum 1 fix-and-recheck cycle.** If the re-validation after a fix still fails, stop and report. Do not attempt further fixes.
- **Never stack fixes.** Apply at most one fix per cycle.
- **Preserve checker isolation.** The checker skill must remain read-only. All code changes go through the fixer skill.
- **Clean work directory** between validation runs: delete `/tmp/genai-model-check/model_ir` only if re-export is needed. Reuse the exported model if the fix was only in tooling (llm_bench/wwb).

## Behavioral Rules

- Follow each sub-skill's behavioral rules strictly
- Do not improvise fixes outside the fixer skill's scope (tools/ directory only)
- Do not modify test infrastructure, CI configuration, or the checker/fixer skills themselves
- If the model export succeeded in Phase 1, reuse the exported IR in Phase 3 (no re-export needed)

## Security Rules

- All security rules from both sub-skills apply
- **NEVER** install any packages
- **NEVER** pass `--trust-remote-code` to any command
- **NEVER** modify the model_id
