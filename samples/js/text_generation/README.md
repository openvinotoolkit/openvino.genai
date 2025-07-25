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
If a converted model in OpenVINO IR format is already available in the collection of [OpenVINO optimized LLMs](https://huggingface.co/collections/OpenVINO/llm-6687aaa2abca3bbcec71a9bd) on Hugging Face, it can be downloaded directly via huggingface-cli.
```sh
pip install huggingface-hub
huggingface-cli download <model> --local-dir <output_folder>
```

### Using GGUF models

To run any samples with a GGUF model, simply provide the path to the .gguf file in the `model_dir` parameter.

This capability is currently available in preview mode and supports a limited set of topologies, including SmolLM, Qwen2.5. For other models 
and architectures, we still recommend converting the model to the IR format, using the optimum-intel tool.

## Sample Descriptions
### Common information

Compile GenAI JavaScript bindings archive first using the instructions in [../../../src/js/README.md](../../../src/js/README.md#build-bindings).

Run `npm install` and the examples will be ready to run.

Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM. For example, the model meta-llama/Llama-2-13b-chat-hf can benefit from being run on a dGPU. Modify the source code to change the device for inference to the GPU.

See https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md#supported-models for the list of supported models.

### 1. Chat Sample (`chat_sample`)
- **Description:**
Interactive chat interface powered by OpenVINO.
Recommended models: meta-llama/Llama-2-7b-chat-hf, TinyLlama/TinyLlama-1.1B-Chat-v1.0, etc
- **Main Feature:** Real-time chat-like text generation.
- **Run Command:**
  ```bash
  node chat_sample.js model_dir
  ```
#### Missing chat template
If you encounter an exception indicating a missing "chat template" when launching the `ov::genai::LLMPipeline` in chat mode, it likely means the model was not tuned for chat functionality. To work this around, manually add the chat template to tokenizer_config.json of your model.
The following template can be used as a default, but it may not work properly with every model:
```
"chat_template": "{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n<|im_start|>assistant\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|im_end|>\n'}}{% endif %}{% endfor %}",
```

### 2. Greedy Causal LM (`greedy_causal_lm`)
- **Description:**
Basic text generation using a causal language model.
Recommended models: meta-llama/Llama-2-7b-hf, etc
- **Main Feature:** Demonstrates simple text continuation.
- **Run Command:**
  ```bash
  node greedy_causal_lm.js model_dir prompt
  ```

### 3. Beam Search Causal LM (`beam_search_causal_lm`)
- **Description:**
Uses beam search for more coherent text generation.
Recommended models: meta-llama/Llama-2-7b-hf, etc
- **Main Feature:** Improves text quality with beam search.
- **Run Command:**
  ```bash
  node beam_search_causal_lm.js model_dir prompt [prompts ...]
  ```

### 4. Multinomial Causal LM (`multinomial_causal_lm`)
- **Description:** Text generation with multinomial sampling for diversity.
Recommended models: meta-llama/Llama-2-7b-hf, etc
- **Main Feature:** Introduces randomness for creative outputs.
- **Run Command:**
  ```bash
  node multinomial_causal_lm.js model_dir prompt
  ```

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
