# OpenVINO GenAI Text Generation C Samples

## Table of Contents
1. [Download OpenVINO GenAI](#download-openvino-genai)
2. [Build Samples](#build-samples)
3. [Download and Convert the Model and Tokenizers](#download-and-convert-the-model-and-tokenizers)
4. [Sample Descriptions](#sample-descriptions)
5. [Support and Contribution](#support-and-contribution)

## Download OpenVINO GenAI

Download and extract [OpenVINO GenAI Archive](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?PACKAGE=OPENVINO_GENAI&VERSION=NIGHTLY&OP_SYSTEM=WINDOWS&DISTRIBUTION=ARCHIVE) Visit the OpenVINO Download Page.


## Build Samples
Set up the environment and build the samples Linux and macOS
```sh
source <INSTALL_DIR>/setupvars.sh
./<INSTALL_DIR>/samples/c/build_samples.sh
```
Windows Command Prompt:
```sh
<INSTALL_DIR>\setupvars.bat
<INSTALL_DIR>\samples\c\build_samples_msvc.bat
```
Windows PowerShell
```sh
.<INSTALL_DIR>\setupvars.ps1
.<INSTALL_DIR>\samples\c\build_samples.ps1
```

## Download and convert the model and tokenizers
The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.
Install [../../export-requirements.txt](../../export-requirements.txt) if model conversion is required.
```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
optimum-cli export openvino --model <model> <output_folder>
```
If a converted model in OpenVINO IR format is available in the [OpenVINO optimized LLMs](https://huggingface.co/collections/OpenVINO/llm-6687aaa2abca3bbcec71a9bd) collection on Hugging Face, you can download it directly via huggingface-cli.
```sh
pip install huggingface-hub
huggingface-cli download <model> --local-dir <output_folder>
```

### Using GGUF models

To run any samples with a GGUF model, simply provide the path to the .gguf file via the `model_dir` parameter.

This capability is currently available in preview mode and supports a limited set of topologies, including SmolLM and Qwen2.5. For other models 
and architectures, we still recommend converting the model to the IR format using the `optimum-intel` tool.

### Sample Descriptions

#### Chat Sample (`chat_sample_c`)
Multi-turn conversations with an interactive chat interface powered by OpenVINO.
- **Run Command:**
```sh
./chat_sample_c model_dir
```

#### LLMs benchamrking sample(`benchmark_genai_c`)
The sample demonstrates how to benchmark LLMs in OpenVINO GenAI by using C language. 
- **Run Command:**
```sh
./benchmark_gena_c  [-m MODEL] [-p PROMPT] [-nw NUM_WARMUP] [-n NUM_ITER] [-mt MAX_NEW_TOKENS] [-d DEVICE]
```
- **Options:**
- `-m, --model`: Path to the model and tokenizers base directory.
- `-p, --prompt` (default: `"The Sky is blue because"`): The prompt to generate text.
- `-nw, --num_warmup` (default: `1`): Number of warmup iterations.
- `-mt, --max_new_tokens` (default: `20`): Maximal number of new tokens.
- `-n, --num_iter` (default: `3`): Number of iterations.
- `-d, --device` (default: `"CPU"`): Device to run the model on.


#### Greedy Causal LM(`greedy_causal_lm`)

Basic text generation using a causal language model. 
- **Run Command:**
```sh
./greedy_causal_lm_c  model_dir prompt
```


## Support and Contribution
- For troubleshooting, consult the [OpenVINO documentation](https://docs.openvino.ai).
- To report issues or contribute, visit the [GitHub repository](https://github.com/openvinotoolkit/openvino.genai).

