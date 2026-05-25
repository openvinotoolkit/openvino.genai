---
name: wwb-fail-analyzer
description: "Analyze results of WWB execution with a specific model with all possible backends. Use when: checking wwb logs, troubleshooting model inference or accuracy issues."
argument-hint: "model_id and log_dir (e.g. tencent/HY-MT1.5-1.8B /path/to/folder/with/logs/wwb/)"
---

# WWB Fail Analyzer

Analyzes failed WWB execution results for a HuggingFace model or a model converted with OpenVINO and inference with GenAI or optimum-intel and provides insights for troubleshooting. It is recommended to run .github/skills/model-checker/SKILL.md first.

## When to Use

- WWB fails during model validation in the model enablement workflow

## Inputs

The user must provide:

- **model_id**: HuggingFace model identifier (e.g. `tencent/HY-MT1.5-1.8B`)
- **log_dir**: path to the folder containing the failed WWB log files

## Code Structure Reference

When analyzing failures and implementing fixes, refer to the following key locations in the codebase:

**Main entry point**:
- `tools/who_what_benchmark/whowhatbench/wwb.py` - argument parsing, main execution flow, function for generation with GenAI

**Model loading and pipeline creation**:
- `tools/who_what_benchmark/whowhatbench/model_loaders.py` - model loading for HF, optimum-intel and GenAI backends

**Generation and evaluation**:
- `tools/who_what_benchmark/whowhatbench/text_evaluator.py` - llm models evaluation, evaluator for --model-type text
- `tools/who_what_benchmark/whowhatbench/visualtext_evaluator.py` - vlm models evaluation, evaluator for --model-type visual-text
- `tools/who_what_benchmark/whowhatbench/chat_text_evaluator.py` - llm models evaluation in chat mode, evaluator for --model-type text-chat
- `tools/who_what_benchmark/whowhatbench/chat_visualtext_evaluator.py` - vlm models evaluation in chat mode, evaluator for --model-type visual-text-chat
- `tools/who_what_benchmark/whowhatbench/embeddings_evaluator.py` - embedding models evaluation, evaluator for --model-type text-embedding
- `tools/who_what_benchmark/whowhatbench/im2im_evaluator.py` - image-to-image evaluation, evaluator for --model-type image-to-image
- `tools/who_what_benchmark/whowhatbench/inpaint_evaluator.py` - inpainting evaluation, evaluator for --model-type image-inpainting
- `tools/who_what_benchmark/whowhatbench/reranking_evaluator.py` - reranking evaluation, evaluator for --model-type text-reranking
- `tools/who_what_benchmark/whowhatbench/speech_generation_evaluator.py` - speech generation evaluation, evaluator for --model-type speech-generation
- `tools/who_what_benchmark/whowhatbench/text2image_evaluator.py` - text-to-image evaluation, evaluator for --model-type text-to-image
- `tools/who_what_benchmark/whowhatbench/text2video_evaluator.py` - text-to-video evaluation, evaluator for --model-type text-to-video

**Similarity measurements**:
  `tools/who_what_benchmark/whowhatbench/whowhat_metrics.py` - classes and functions for calculating similarity
  `tools/who_what_benchmark/whowhatbench/tts_similarity.py` - tool for calculating similarity for audio

**Utility functions**:
- `tools/who_what_benchmark/whowhatbench/utils.py` - various utility functions for logging, error handling, file operations, etc.
- `tools/who_what_benchmark/whowhatbench/inputs_preprocessors` - preprocessing of input data for VLM models

**Use this reference throughout Steps 1-3 when analyzing logs and identifying where to implement fixes.**


## Procedure

### Step 1: Analyze the log for HuggingFace backend

If the log file log for ground truth generation with hf backend doesn't contains failure (exit code == 0 and no error in the output), process to Step 2. Otherwise, follow the next steps:
  - Read the corresponding log for the full traceback and context.
  - Analyze the failure root cause base on the log and Code Structure Reference.
  - Determine whether the error is related to the WWB or to the environment/backend limitations.
  - If it's a WWB bug/limitation, implement nessesary fixes to WWB. Use the Code Structure Reference to locate the exact functions to modify. Follow OpenVINO GenAI coding guidelines from `.github/copilot-instructions.md`. Ensure changes don't break existing functionality. Add appropriate error messages and logging. Test changes by re-running model-checker with `--skip-export` and `--skip-llm_bench` flags.
  - If it's a model issue or backend limitation, provide description in the report.

**Most common failure points**:
- Model is not supported with current transformers version. If transformers version necessary for the model is higher than version in requirements.txt, update requirements.txt. Otherwise, reflect the recommendation to install of an earlier version in the report. Do not install any packages.
- Some modules is not installed in the environment. Add this module to `requirements.txt` or the appropriate dependency/constraints file, and describe the required dependency update in the report; do not install packages in the analysis environment.
- WWB use wrong model class or arguments for model loading/generation. Analyze and find appropriate class or args for the model and add them to `model_loaders.py` or corresponding evaluator. Don't remove existing functionality, only add new conditions for the new model architecture or features.
- If `--model-type` is `visual-text` or `visual-text-chat`, check if the input preprocessing in `inputs_preprocessors` folder exists for coresponding model_id. If not, add necessary preprocessing steps. Investigate the file https://github.com/huggingface/optimum-intel/blob/main/optimum/intel/openvino/modeling_visual_language.py to find out how function `preprocess_inputs` are established for other models in the optimum-intel; perhaps you will find a function there for the current model. Based on the function example and examples of other classes from the `inputs_preprocessors` folder, create a new class for the current model_id. `preprocess_inputs` function should contains functionality for question answering and chat cases.

### Step 2: Analyze the log for GenAI backend

If the log file for GenAI backend doesn't contains failure (exit code == 0 and no error in the output) and trashhold is high, process to Step 3. Otherwise, follow the next steps:
  - Read the corresponding log for the full traceback and context.
  - Analyze the failure root cause from the log. Don't analyze GenAI source code.
  - Determine whether the error is related to the WWB or to the environment/backend limitations.
  - If it's a WWB bug/limitation, implement nessesary fixes to WWB. Use the Code Structure Reference to locate the exact functions to modify. Follow OpenVINO GenAI coding guidelines from `.github/copilot-instructions.md`. Ensure changes don't break existing functionality. Add appropriate error messages and logging. Test changes by re-running model-checker with `--skip-export`, `--skip-wwb-gt-data-gen` and `--skip-llm-bench` flags.
  - If it's a GenAI limitation or model issue, provide description in the report.

**Most common failure points**:
- Model is not supported by the GenAI. If the model is not supported, report it as a GenAI limitation.

### Step 3: Analyze the log for optimum-intel backend

If the log file for optimum-intel backend doesn't contains failure (exit code == 0 and no error in the output) and trashhold is high, process to Step 4. Otherwise, follow the next steps:
  - Read the corresponding log for the full traceback and context.
  - Analyze the failure root cause from the log.
  - Determine whether the error is related to the WWB or to the environment/backend limitations.
  - If it's a WWB bug/limitation, implement nessesary fixes to WWB. Use the Code Structure Reference to locate the exact functions to modify. Follow OpenVINO GenAI coding guidelines from `.github/copilot-instructions.md`. Ensure changes don't break existing functionality. Add appropriate error messages and logging. Test changes by re-running model-checker with `--skip-export`, `--skip-wwb-gt-data-gen` and `--skip-llm-bench` flags.
  - If it's a model issue or backend limitation, provide description in the report.

**Most common failure points**:
- WWB use wrong model class or argument list. Analyze optimum-intel module, find appropriate class and argument for the model loading/generation and update WWB source code. Don't remove existing functionality in model_loaders.py, only add new conditions for the new model architecture or features.
- If `--model-type` is `visual-text-chat`, check if the input preprocessing in `inputs_preprocessors` folder exists for coresponding model_id. If not, add necessary preprocessing steps. Investigate the file https://github.com/huggingface/optimum-intel/blob/main/optimum/intel/openvino/modeling_visual_language.py to find out how function `preprocess_inputs` are established for other models in the optimum-intel; perhaps you will find a function there for the current model. Based on the function example and examples of other classes from the `inputs_preprocessors` folder, create a new class for the current model_id. `preprocess_inputs` function should contains functionality for question answering and chat cases.
- If threshold is low, check whether chat_template is correctly applied and the model output correctly processed in the evaluator.

### Step 4: Report Results

Results format:

## Model Information
- **Model ID**: `<model_id>`
- **Task**: `<task>`

## Backend Analysis Summary

### HuggingFace Backend
- **Status**: PASSED / FAILED
- **Log**: `<log_dir>/wwb_hf_ground_truth.log`
- **Issue**: <description if failed>
- **Error message**: <relevant error messages if failed>
- **Possible fix**: <description of fix if failed>
- **WWB modification**: WWB was modified / nothing changed
- **After fix status**: <PASSED / FAILED if failed>

### Optimum Intel Backend
- **Status**: PASSED / FAILED
- **Log**: `<log_dir>/wwb_optimum_target_eval.log`
- **Similarity**: <value if available>
- **Issue**: <description if failed>
- **Error message**: <relevant error messages if failed>
- **Possible fix**: <description of fix if failed>
- **WWB modification**: WWB was modified / nothing changed
- **After fix status**: <PASSED / FAILED if failed>

### GenAI Backend
- **Status**: PASSED / FAILED
- **Log**: `<log_dir>/wwb_genai_target_eval.log`
- **Similarity**: <value if available>
- **Issue**: <description if failed>
- **Error message**: <relevant error messages if failed>
- **Possible fix**: <description of fix if failed>
- **WWB modification**: WWB was modified / nothing changed
- **After fix status**: <PASSED / FAILED if failed>

### Step 5: Create Pull Request (if fixes were made)

**Condition**: Only proceed if code changes were made in Step 1 or Step 2 or Step 3.

Read and follow the **open-pr** skill located at `.github/skills/open-pr/SKILL.md`.

**PR details**:
- **Title**: `[wwb] Add fix for <model_id>`
- **Description**: Include:
  - Summary of changes made
  - Testing performed (re-run results with model-checker)
  - Impact on existing functionality

If no fixes were made (e.g., backend limitation identified), skip this step and note in the report that architectural work is needed.

### Security

- **NEVER** install any packages. Assume the environment is pre-configured.
- **NEVER** modify `model_id` — pass it exactly as provided by the user.
