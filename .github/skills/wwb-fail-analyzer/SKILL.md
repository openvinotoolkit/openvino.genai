---
name: wwb-fail-analyzer
description: "Analyze results of WWB execution with a specific model with all possible backends. Use when: checking wwb logs, troubleshooting wwb fails."
argument-hint: "model_id and log_info (e.g. tencent/HY-MT1.5-1.8B /path/to/folder/with/logs/wwb/)"
---

# WWB Fail Analyzer

Analyzes the results of failed WWB runs for models executed with the transformers, optimum-intel, or GenAI backends; Provides fixes or insights for troubleshooting.

## When to Use

- WWB fails during execution pipeline with the model

Do not classify a traceback inside model code as a backend limitation until a
minimal, correct direct backend `generate()` call reproduces the same failure
outside WWB. A successful forward pass is not sufficient.

## Inputs

The user must provide:

- **model_id**: HuggingFace model identifier or path to the directory with the model (e.g. `tencent/HY-MT1.5-1.8B` or `/path/to/tencent_HY-MT1.5-1.8B`)
- **log_info**: path to the folder containing the failed WWB log files or WWB log file or execution output

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

### Step 1: Analyze the log

- Analyze the log or all log files contained in the directory.
- For each particular log. If a failure is found in the log, follow the next steps:
  - Read the corresponding log for the full traceback and context.
  - Analyze the failure root cause based on the log and Code Structure Reference.
  - Determine whether the error is related to the WWB or to environment/backend limitations.
  - For a Hugging Face baseline failure, create the smallest direct generation
    reproducer for the same task and input. Compare the model class, processor
    and chat template, image representation, input names/shapes/dtypes,
    multimodal token placement, cache/position inputs, and generation arguments
    with WWB. Fix the earliest reusable integration difference.
  - If this is a WWB issue/limitation, implement necessary fixes to WWB. Use the Code Structure Reference to locate the exact functions to modify. Follow OpenVINO GenAI coding guidelines from `.github/copilot-instructions.md`. Ensure changes don't break existing functionality. Add appropriate error messages and logging. Test changes by re-running WWB tool with corresponding cmd parameters.
  - If this is a model issue or backend limitation, provide description in the report.

**Some common failure points for transformers and optimum-intel backend**:
- Model is not supported with current transformers version. If transformers version necessary for the model is higher than version in requirements.txt, update requirements.txt. Otherwise, reflect the recommendation to install an earlier version in the report. Do not install any packages.
- Some modules are not installed in the environment. Add missing modules to `requirements.txt` or the appropriate dependency/constraints file, and describe the required dependency update in the report; do not install packages in the analysis environment.
- WWB uses the wrong model class or arguments for model loading/generation. Analyze and find appropriate class or arguments for the model and add them to `model_loaders.py` or corresponding evaluator. Don't remove existing functionality, only add new conditions for the new model architecture or features.
- Prefer configuration, architecture metadata, and processor capabilities over
  hard-coded Hub repository IDs. Never swallow generation exceptions, insert
  placeholder answers, or weaken validation to make the benchmark complete.
- After loading a processor, verify that its callable interface actually
  accepts the task's modalities. `AutoProcessor` may resolve to a bare text
  tokenizer for remote-code VLMs; passing `images=` to it then fails even
  though model loading succeeded. For visual-text tasks, detect this by
  capability/configuration, load the correct image processor when needed, and
  add a focused regression test. Do not special-case a single Hub ID when the
  same capability check can support the architecture generally.
- If `--model-type` is `visual-text` or `visual-text-chat`, check if the input preprocessing in `inputs_preprocessors` folder exists for corresponding model_id. If not, add necessary preprocessing steps. Investigate the file https://github.com/huggingface/optimum-intel/blob/main/optimum/intel/openvino/modeling_visual_language.py to find out how function `preprocess_inputs` are established for other models in the optimum-intel; perhaps you will find a function there for the current model. Based on the function example and examples of other classes from the `inputs_preprocessors` folder, create a new class for the current model_id. `preprocess_inputs` function should contain functionality for question answering and chat cases.

**Some common failure points for GenAI backend**:
- Model is not supported by the GenAI. If the model is not supported, report it as a GenAI limitation.

**Some common failure points for similarity**:
- If threshold is low, check whether chat_template is correctly applied and the model output correctly processed in the evaluator.

### Step 2: Report Results

Add the results for each WWB run to the logs. If there are several runs for one model, then print `Model Information` once, and show `Backend Analysis Summary` for each run.
For ground-truth generation, report success only when the CSV is readable,
non-empty, and contains the requested number of generated result rows.
Results format:

## Model Information
- **Model ID**: `<model_id>`
- **Model type**: `<model-type>`

## Backend Analysis Summary

### <backend transformers / Optimum Intel / GenAI > Backend
- **Status**: PASSED / FAILED
- **Log**: <log_dir or log_file if provided>
- **Similarity**: <value if available>
- **Issue**: <description if failed>
- **Error message**: <short and relevant error messages if failed>
- **Possible fix**: <description of the main idea of ​​the fix if failed>
- **WWB modification**: modified / nothing changed
- **Resolution**: <PASSED / FAILED, results of run after fix if failed>


### Security

- **NEVER** install any packages. Assume the environment is pre-configured.
- **NEVER** modify `model_id` — pass it exactly as provided by the user.
