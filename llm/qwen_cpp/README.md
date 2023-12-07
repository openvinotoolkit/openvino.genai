# LLM

This application showcases inference of a large language model (LLM). It doesn't have much of configuration options to encourage the reader to explore and modify the source code. There's a Jupyter notebook which corresponds to this pipeline and discusses how to create an LLM-powered Chatbot: https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot.

> Note  
This pipeline is not for production use.

## How it works

The program loads a tokenizer and a model (`.xml` and `.bin`) to OpenVINO™. The model is reshaped to batch 1 and variable prompt length. A prompt is tokenized and passed to the model. The model greedily generates token by token until the special end of sequence (EOS) token is obtained. The predicted tokens are converted to chars and printed in a streaming fashion.

## Supported models

1. Qwen
    1. https://huggingface.co/Qwen/Qwen-7B
    2. https://huggingface.co/Qwen/Qwen-7B-Chat
    3. https://huggingface.co/Qwen/Qwen-7B-Chat-Int4

## Qwen Demo
### Step 1: Download and convert the model

```sh
cd openvino.genai/llm_bench/python
python -m pip install -r requirements.txt
python convert.py --model_id Qwen/Qwen-7B-Chat-Int4 --output_dir Qwen-7B-Chat-GPTQ_INT4_FP16-2K --precision FP16
```

### Step 2: [Download archives](https://storage.openvinotoolkit.org/repositories/openvino/packages/) with latest OpenVINO™ release

### Step 3: Build Qwen CPP pipeline
```sh
cd openvino.genai/llm/cpp/qwen_cpp
git submodule update --init --recursive
<OpenVINO dir>/setupvars.bat
cmake -B build
cmake --build build -j --config Release
```

### Step 4: Run LLM Chat with Qwen Model via OpenVINO runtime
### Run on Windows
Usage: `.\build\bin\Release\main.exe -m <openvino_model.xml> -t <qwen.tiktoken> -d <device>"`

Example: `.\build\bin\Release\main.exe -m Qwen-7B-Chat-GPTQ_INT4_FP16-2K\openvino_model.xml -t Qwen-7B-Chat-GPTQ_INT4_FP16-2K\qwen.tiktoken -d CPU`

### Run on Linux
Usage: `./build/bin/main -m <openvino_model.xml> -t <qwen.tiktoken> -d <device>"`

Example: `./build/bin/main -m Qwen-7B-Chat-GPTQ_INT4_FP16-2K/openvino_model.xml -t Qwen-7B-Chat-GPTQ_INT4_FP16-2K/qwen.tiktoken -d CPU`
