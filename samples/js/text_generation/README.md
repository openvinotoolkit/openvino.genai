# JavaScript chat_sample that supports most popular models like LLaMA 3

This example showcases inference of text-generation Large Language Models (LLMs): `chatglm`, `LLaMA`, `Qwen` and other models with the same signature. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample fearures `Pipeline.LLMPipeline` and configures it for the chat scenario.

## Download and convert the model and tokenizers

To convert model you have to use python package `optimum-intel`.
The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
optimum-cli export openvino --trust-remote-code --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
```

## Run:

Compile GenAI JavaScript bindings archive first using the instructions in [../../../src/js/README.md](../../../src/js/README.md#build-bindings).

Run `npm install` in current folder and then run the sample:

`node chat_sample.js TinyLlama-1.1B-Chat-v1.0`

Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM. For example, the model meta-llama/Llama-2-13b-chat-hf can benefit from being run on a dGPU. Modify the source code to change the device for inference to the GPU.

See https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md#supported-models for the list of supported models.

### Troubleshooting

#### Unicode characters encoding error on Windows

Example error:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u25aa' in position 0: character maps to <undefined>
```

If you encounter the error described in the example when sample is printing output to the Windows console, it is likely due to the default Windows encoding not supporting certain Unicode characters. To resolve this:
1. Enable Unicode characters for Windows cmd - open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.
2. Enable UTF-8 mode by setting environment variable `PYTHONIOENCODING="utf8"`.

#### Missing chat template

If you encounter an exception indicating a missing "chat template" when launching the `ov::genai::LLMPipeline` in chat mode, it likely means the model was not tuned for chat functionality. To work this around, manually add the chat template to tokenizer_config.json of your model.
The following template can be used as a default, but it may not work properly with every model:
```
"chat_template": "{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n<|im_start|>assistant\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|im_end|>\n'}}{% endif %}{% endfor %}",
```
