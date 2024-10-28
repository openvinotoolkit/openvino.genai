# Text generation Python greedy_causal_lm that supports most popular models like LLaMA 3

This example showcases inference of text-generation Large Language Models (LLMs): `chatglm`, `LLaMA`, `Qwen` and other models with the same signature. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample fearures `openvino_genai.LLMPipeline` and configures it to run the simplest deterministic greedy sampling algorithm. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-chatbot) which provides an example of LLM-powered Chatbot in Python.

There are two sample files:
 - [`greedy_causal_lm.py`](./greedy_causal_lm.py) demonstrates basic usage of the LLM pipeline
 - [`lora.py`](./lora.py) shows how to apply LoRA adapters to the pipeline

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

It's not required to install [../../export-requirements.txt](../../export-requirements.txt) for deployment if the model has already been exported. [../../deployment-requirements.txt](../../deployment-requirements.txt) describes the requirements to deploy all samples. [../../requirements.txt](../../requirements.txt) includes packages for both scenarios.

```sh
pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --trust-remote-code --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
```

## Run

[../../deployment-requirements.txt](../../deployment-requirements.txt) describes the requirements to deploy all samples. [../../requirements.txt](../../requirements.txt) includes packages for both scenarios: converting models and deployment.

`python greedy_causal_lm.py TinyLlama-1.1B-Chat-v1.0 "Why is the Sun yellow?"`


Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM. For example, the model meta-llama/Llama-2-13b-chat-hf can benefit from being run on a dGPU. Modify the source code to change the device for inference to the GPU.

See https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md#supported-models for the list of supported models.

## Run with optional LoRA adapters

LoRA adapters can be connected to the pipeline and modify generated text. Adapters are supported in Safetensors format and can be downloaded from public sources like [Civitai](https://civitai.com) or [HuggingFace](https://huggingface.co/models) or trained by the user. Adapters compatible with a base model should be used only. A weighted blend of multiple adapters can be applied by specifying multiple adapter files with corresponding alpha parameters in command line. Check `lora.py` source code to learn how to enable adapters and specify them in each `generate` call.

Here is an example how to run the sample with a single adapter. First download adapter file from TODO page manually and save it as TODO. Or download it from command line:

#TODO command to download adapter

Then run `lora.py`:

#TODO command to run lora.py with adapter

### Troubleshooting

#### Unicode characters encoding error on Windows

Example error:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u25aa' in position 0: character maps to <undefined>
```

If you encounter the error described in the example when sample is printing output to the Windows console, it is likely due to the default Windows encoding not supporting certain Unicode characters. To resolve this:
1. Enable Unicode characters for Windows cmd - open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.
2. Enable UTF-8 mode by setting environment variable `PYTHONIOENCODING="utf8"`.
