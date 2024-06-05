# C++ chat_sample that supports most popular models like LLaMA 2

This example showcases inference of text-generation Large Language Models (LLMs): `chatglm`, `LLaMA`, `Qwen` and other models with the same signature. The application don't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample fearures `ov::genai::LLMPipeline` and configures it for the chat scenario. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot) which provides an example of LLM-powered Chatbot in Python.

## Install OpenVINO

Install [OpenVINO Archives >= 2024.2](docs.openvino.ai/install). `master` and possibly the latest `releases/*` branch correspond to not yet released OpenVINO versions. https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/ can be used for these branches early testing. `<INSTALL_DIR>` below refers to the extraction location.

## Install OpenVINOGenAI

Follow [../../../src/README.md](../../../src/README.md).

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

#### Linux/macOS

```sh
source <INSTALL_DIR>/setupvars.sh
python3 -m pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --trust-remote-code --weight-format fp16 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
```

#### Windows

```bat
<INSTALL_DIR>\setupvars.bat
python -m pip install --upgrade-strategy eager -r requirements.txt
optimum-cli export openvino --trust-remote-code --weight-format fp16 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
```

## Run

### Usage:
`chat_sample <MODEL_DIR>`

### Examples:

#### Linux/MacOS:
`./build/samples/cpp/chat_sample/chat_sample ./TinyLlama-1.1B-Chat-v1.0/`

#### Windows:
`.\build\samples\cpp\chat_sample\Release\chat_sample .\TinyLlama-1.1B-Chat-v1.0\`

To enable Unicode characters for Windows cmd open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.

Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM. For example, the model meta-llama/Llama-2-13b-chat-hf can benefit from being run on a dGPU. Modify the source code to change the device for inference to the GPU.

See [../../../src/README.md#supported-models](../../src/README.md#supported-models) for the list of supported models.
