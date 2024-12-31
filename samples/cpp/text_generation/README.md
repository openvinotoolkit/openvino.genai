# Text generation C++ greedy_causal_lm that supports most popular models like LLaMA 3

This example showcases inference of text-generation Large Language Models (LLMs): `chatglm`, `LLaMA`, `Qwen` and other models with the same signature. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample fearures `ov::genai::LLMPipeline` and configures it to run the simplest deterministic greedy sampling algorithm. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-chatbot) which provides an example of LLM-powered Chatbot in Python.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

It's not required to install [../../export-requirements.txt](../../export requirements.txt) for deployment if the model has already been exported.

```sh
pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --trust-remote-code --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
```

## Run

Follow [Get Started with Samples](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/get-started-demos.html) to run the sample.

`greedy_causal_lm TinyLlama-1.1B-Chat-v1.0 "Why is the Sun yellow?"`


Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM. For example, the model meta-llama/Llama-2-13b-chat-hf can benefit from being run on a dGPU. Modify the source code to change the device for inference to the GPU.

See https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md#supported-models for the list of supported models.

## Using encrypted models

LLMPipeline and Tokenizer objects can be initialized directly from the memory buffer, e.g. when user stores only encrypted files and decrypts them on-the-fly. 
The following code snippet demonstrates how to load the model from the memory buffer:

```cpp
auto [model_str, weights_tensor] = decrypt_model(models_path + "/openvino_model.xml", models_path + "/openvino_model.bin");
ov::genai::Tokenizer tokenizer(models_path);
ov::genai::LLMPipeline pipe(model_str, weights_tensor, tokenizer, device);
```
For the sake of brevity the code above does not include Tokenizer decryption. For more details look to encrypted_model_causal_lm sample.

### Troubleshooting

#### Unicode characters encoding error on Windows

Example error:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u25aa' in position 0: character maps to <undefined>
```

If you encounter the error described in the example when sample is printing output to the Windows console, it is likely due to the default Windows encoding not supporting certain Unicode characters. To resolve this:
1. Enable Unicode characters for Windows cmd - open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.
2. Enable UTF-8 mode by setting environment variable `PYTHONIOENCODING="utf8"`.
