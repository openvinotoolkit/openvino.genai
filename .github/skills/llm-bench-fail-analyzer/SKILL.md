---
name: llm-bench-fail-analyzer
description: "Analyze failed llm_bench execution results for a model. Use when: checking llm_bench log, troubleshooting model inference or performance issues."
argument-hint: "model_id and log_dir (e.g. tencent/HY-MT1.5-1.8B /path/to/logs/for/llm_bench)"
---

# LLM Bench Fail Analyzer

Analyzes failed llm_bench results for a model converted with OpenVINO and inference with GenAI and provides insights for troubleshooting.

## When to Use

- llm_bench fails during model validation in the model enablement workflow

## Inputs

The user must provide:

- **model_id**: HuggingFace model identifier (e.g. `tencent/HY-MT1.5-1.8B`)
- **log_dir**: path to the folder containing the failed llm_bench log files

## Code Structure Reference

When analyzing failures and implementing fixes, refer to the following key locations in the codebase:

**Main benchmark script**:
- `tools/llm_bench/benchmark.py` - Entry point for benchmarking, command-line argument parsing, and task orchestration

**Task implementations** (task/):
- `tools/llm_bench/task/text_generation.py` - Text generation benchmarking for LLMs
- `tools/llm_bench/task/image_generation.py` - Image generation benchmarking for diffusion models
- `tools/llm_bench/task/visual_language_generation.py` - VLM benchmarking for multimodal models
- `tools/llm_bench/task/video_generation.py` - Video generation benchmarking
- `tools/llm_bench/task/super_resolution_generation.py` - Super resolution benchmarking
- `tools/llm_bench/task/speech_to_text_generation.py` - Speech-to-text (ASR) benchmarking
- `tools/llm_bench/task/text_to_speech_generation.py` - Text-to-speech (TTS) benchmarking
- `tools/llm_bench/task/text_embeddings.py` - Text embedding model benchmarking
- `tools/llm_bench/task/text_reranker.py` - Text reranking model benchmarking
- `tools/llm_bench/task/pipeline_utils.py` - Common pipeline utilities and base classes for all tasks

**Core utilities** (llm_bench_utils/):
- `tools/llm_bench/llm_bench_utils/config_class.py` - Configuration classes, model class definitions, attention backend settings
- `tools/llm_bench/llm_bench_utils/model_utils.py` - Model utility functions: parameter loading, config parsing, precision handling
- `tools/llm_bench/llm_bench_utils/ov_utils.py` - OpenVINO model creation and management (GenAI, optimum-intel)
- `tools/llm_bench/llm_bench_utils/pt_utils.py` - PyTorch model creation and torch.compile support
- `tools/llm_bench/llm_bench_utils/ov_model_classes.py` - Custom OpenVINO model classes (OVMPTModel, OVChatGLMModel, etc.)
- `tools/llm_bench/llm_bench_utils/prompt_utils.py` - Prompt loading, preprocessing for text/image/video inputs
- `tools/llm_bench/llm_bench_utils/parse_json_data.py` - JSON data parsing for prompts and configurations
- `tools/llm_bench/llm_bench_utils/get_use_case.py` - Use case detection and configuration

**Performance measurement** (llm_bench_utils/):
- `tools/llm_bench/llm_bench_utils/hook_forward.py` - Forward pass hooks for performance measurement
- `tools/llm_bench/llm_bench_utils/hook_greedy_search.py` - Greedy search hooks
- `tools/llm_bench/llm_bench_utils/hook_beam_search.py` - Beam search hooks
- `tools/llm_bench/llm_bench_utils/hook_common.py` - Common hook utilities
- `tools/llm_bench/llm_bench_utils/hook_forward_whisper.py` - Whisper-specific forward hooks

**Output and reporting** (llm_bench_utils/):
- `tools/llm_bench/llm_bench_utils/metrics_print.py` - Metrics printing and logging to console
- `tools/llm_bench/llm_bench_utils/output_json.py` - JSON output generation
- `tools/llm_bench/llm_bench_utils/output_csv.py` - CSV output generation
- `tools/llm_bench/llm_bench_utils/output_file.py` - General file output utilities
- `tools/llm_bench/llm_bench_utils/gen_output_data.py` - Generated output data handling

**Hook implementations** (llm_bench_utils/llm_hook_sample/ and llm_hook_beam_search/):
- Version-specific hook implementations for different OpenVINO versions (v4_43, v4_45, v4_51, v4_52, v4_55, v4_57, v5, v5_3)
- `tools/llm_bench/llm_bench_utils/llm_hook_sample/*.py` - Sample generation hooks
- `tools/llm_bench/llm_bench_utils/llm_hook_beam_search/*.py` - Beam search hooks

**Use this reference throughout Steps 1-3 when analyzing logs and identifying where to implement fixes.**

## Procedure


### Step 1: Analyze the logs

If the log file for llm_bench doesn't contains failure (exit code == 0 and no error in the output), proceed to Step 2. Otherwise, follow the next steps:
  - Read the corresponding log for the full traceback and context.
  - Analyze the failure root cause from the log.
  - Define whether fail relates to llm_bench or backend/model.
  - Implement nessesary fixes to llm_bench tool if it's a llm_bench bug/limitation. Use the Code Structure Reference to locate the exact functions to modify. Follow OpenVINO GenAI coding guidelines from `.github/copilot-instructions.md`. Ensure changes don't break existing functionality. Add appropriate error messages and logging. Test changes by re-running model-checker with `--skip-export` and `--skip-wwb` flags.
  - If it's a model issue or backend limitation, provide description in the report.


### Step 2: Report Results

Results format:

## Model Information
- **Model ID**: `<model_id>`
- **Task**: `<task>`
- **Status**: PASSED / FAILED
- **Log**: `<log_dir>/llm_bench.log`
- **Issue**: <description if failed>
- **Error message**: <relevant error messages if failed>
- **Possible fix**: <description of fix if failed>
- **LLM Bench modification**: llm_bench was modified / nothing changed
- **After fix status**: <PASSED / FAILED if failed>

### Step 5: Create Pull Request (if fixes were made)

**Condition**: Only proceed if code changes were made in Step 1 or Step 2 or Step 3.

Read and follow the **open-pr** skill located at `.github/skills/open-pr/SKILL.md`.

**PR details**:
- **Title**: `[llm_bench] Add fix for <model_id>`
- **Description**: Include:
  - Summary of changes made
  - Testing performed (re-run results with model-checker)
  - Impact on existing functionality

If no fixes were made (e.g., backend limitation identified), skip this step and note in the report that architectural work is needed.

### Security

- **NEVER** install any packages. Assume the environment is pre-configured.
- **NEVER** modify `model_id` — pass it exactly as provided by the user.
