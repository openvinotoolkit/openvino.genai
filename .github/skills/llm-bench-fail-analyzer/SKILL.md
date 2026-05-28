---
name: llm-bench-fail-analyzer
description: "Analyze failed llm_bench execution results for a model. Use when: checking llm_bench log, troubleshooting llm_bench fails."
argument-hint: "model_dir and log_info (e.g. /path/to/tencent_HY-MT1.5-1.8B /path/to/logs/for/llm_bench)"
---

# LLM Bench Fail Analyzer

Analyzes failed llm bench results for models, which were run with transformers, optimum-intel or GenAI backends; Provides fixes or insights for troubleshooting.

## When to Use

- llm_bench fails during execution pipeline with the model

## Inputs

The user must provide:

- **model_dir**: path to the directory with the model (e.g. `/path/to/tencent_HY-MT1.5-1.8B`)
- **log_info**: path to the folder containing the failed llm_bench log files or llm_bench log file or execution output

## Code Structure Reference

When analyzing failures and implementing fixes, refer to the following key locations in the codebase:

**Main benchmark script**:
- `tools/llm_bench/benchmark.py` - Entry point for benchmarking, command-line argument parsing, and task orchestration

**Model execution pipeline implementations**:
- `tools/llm_bench/task/text_generation.py` - Text generation benchmarking for LLMs
- `tools/llm_bench/task/image_generation.py` - Image generation(text to image, image to image, inpainting) benchmarking
- `tools/llm_bench/task/visual_language_generation.py` - VLM benchmarking for multimodal models
- `tools/llm_bench/task/video_generation.py` - Video generation benchmarking
- `tools/llm_bench/task/super_resolution_generation.py` - Super resolution benchmarking
- `tools/llm_bench/task/speech_to_text_generation.py` - Speech-to-text (ASR) benchmarking
- `tools/llm_bench/task/text_to_speech_generation.py` - Text-to-speech (TTS) benchmarking
- `tools/llm_bench/task/text_embeddings.py` - Text embedding model benchmarking
- `tools/llm_bench/task/text_reranker.py` - Text reranking model benchmarking
- `tools/llm_bench/task/pipeline_utils.py` - Common pipeline utilities and base classes for all tasks

**Core utilities**:
- `tools/llm_bench/llm_bench_utils/config_class.py` - Configuration classes, model class definitions, attention backend settings
- `tools/llm_bench/llm_bench_utils/model_utils.py` - Model utility functions: parameter loading, config parsing, precision handling
- `tools/llm_bench/llm_bench_utils/ov_utils.py` - OpenVINO model creation and management (GenAI, optimum-intel)
- `tools/llm_bench/llm_bench_utils/pt_utils.py` - PyTorch model creation and torch.compile support
- `tools/llm_bench/llm_bench_utils/ov_model_classes.py` - Custom OpenVINO model classes (OVMPTModel, OVChatGLMModel, etc.)
- `tools/llm_bench/llm_bench_utils/prompt_utils.py` - Prompt loading, preprocessing for text/image/video inputs
- `tools/llm_bench/llm_bench_utils/parse_json_data.py` - JSON data parsing for prompts and configurations
- `tools/llm_bench/llm_bench_utils/get_use_case.py` - Use case detection and configuration

**Model wrappers for performance measurement with transformers/optimum-intel**:
- `tools/llm_bench/llm_bench_utils/hook_forward.py` - hooks for image generation, RAG and TTS pipelines
- `tools/llm_bench/llm_bench_utils/hook_greedy_search.py` - Greedy sampling hooks
- `tools/llm_bench/llm_bench_utils/hook_beam_search.py` - Beam search sampling hooks
- `tools/llm_bench/llm_bench_utils/hook_common.py` - determination of the required hook
- `tools/llm_bench/llm_bench_utils/hook_forward_whisper.py` - ASR forward hook
- `tools/llm_bench/llm_bench_utils/llm_hook_sample/*.py` - version-specific greedy hook implementations for different transformers versions (v4_43, v4_45, v4_51, v4_52, v4_55, v4_57, v5, v5_3)
- `tools/llm_bench/llm_bench_utils/llm_hook_beam_search/*.py` - version-specific beam search hook implementations for different transformers versions (v4_43, v4_45, v4_51, v4_52, v4_55, v4_57, v5, v5_3)

**Output and reporting**:
- `tools/llm_bench/llm_bench_utils/metrics_print.py` - Metrics printing and logging to console
- `tools/llm_bench/llm_bench_utils/output_json.py` - JSON output generation
- `tools/llm_bench/llm_bench_utils/output_csv.py` - CSV output generation
- `tools/llm_bench/llm_bench_utils/output_file.py` - construct file name and save output to file
- `tools/llm_bench/llm_bench_utils/gen_output_data.py` - convert output data to dict format

**Use this reference throughout Steps 1-3 when analyzing logs and identifying where to implement fixes.**


## Procedure


### Step 1: Analyze the logs

If the logs for llm_bench don't contain failures, proceed to Step 2. Otherwise, follow the next steps:
  - Read the corresponding log for the full traceback and context.
  - Analyze the failure root cause from the log.
  - Define whether fail relates to llm_bench or backend/model.
  - If it's a llm_bench bug/limitation, implement necessary fixes to llm_bench tool. Use the Code Structure Reference to locate the exact functions to modify. Follow OpenVINO GenAI coding guidelines from `.github/copilot-instructions.md`. Ensure changes don't break existing functionality. Add appropriate error messages and logging. Test changes by re-running llm_bench tool with corresponding cmd parameters.
  - If it's a model issue or backend limitation, provide description in the report.


### Step 2: Report Results

Add the results for each llm_bench run to the logs.
Results format:

## Model Information
- **Model dir**: `<model_dir>`
- **Task**: `<task>`
- **Status**: PASSED / FAILED
- **Log**: <log_dir or log_file if provided>
- **Issue**: <description if failed>
- **Error message**: <short and relevant error messages if failed>
- **Possible fix**: <description of fix if failed>
- **LLM Bench modification**: modified / nothing changed
- **After fix status**: <PASSED / FAILED if failed>

### Security

- **NEVER** install any packages. Assume the environment is pre-configured.
- **NEVER** modify `model_dir` — pass it exactly as provided by the user.
