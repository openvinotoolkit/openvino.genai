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

To run any samples with a GGUF model, simply provide the path to the .gguf file via the `model_dir` parameter.

This capability is currently available in preview mode and supports a limited set of topologies, including SmolLM and Qwen2.5. For other models 
and architectures, we still recommend converting the model to the IR format using the `optimum-intel` tool.

## Sample Descriptions
### Common information

When you use the [openvino.genai](https://github.com/openvinotoolkit/openvino.genai) **release branch**, install dependencies before running samples.
In the current directory, run:
```bash
npm install
```

If you use the master branch, you may need to follow 
[this instruction](../../../src/js/README.md#build-bindings) 
to build the latest version of `openvino-genai-node` from source first, then install dependencies.


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

### 5. LLM ReAct Agent Sample (`react_sample`)
- **Description:**
Interactive ReAct Agent powered by OpenVINO.
Recommended models: Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct
- **Main Feature:** Real-time reasoning-action from user's input.
- **Run Command:**
  ```bash
  node react_sample.js model_dir
  ```

### 6. LLMs benchmarking sample (`benchmark_genai`)
- **Description:** 
  This sample script demonstrates how to benchmark LLMs in OpenVINO GenAI. The script includes functionality for warm-up iterations, generating text, and calculating various performance metrics.

  For more information on how performance metrics are calculated, please follow the [performance-metrics tutorial](../../../src/README.md#performance-metrics).
- **Main Feature:** Benchmark model via GenAI
- **Run Command:**
  ```bash
  node benchmark_genai.js [-m MODEL] [-p PROMPT] [--nw NUM_WARMUP] [-n NUM_ITER] [--mt MAX_NEW_TOKENS] [-d DEVICE]
  ```

### 7. Structured Output Sample (`structured_output_sample`)
- **Description:**
This sample demonstrates how to use OpenVINO GenAI to generate structured outputs such as JSON from text prompts. This sample implementation is split into multiple "generate" calls to mitigate generating complex, variadic JSON structures in a single pass. This is done because not all models are able to generate a complex JSON, with a variadic number of elements in one shot, especially if the model is small and not fine-tuned for this task. By separating the task into two stages, it becomes possible to use smaller models and still achieve generated JSON good quality.

Recommended models: meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-8B-Instruct
- **Run Command:**
  ```bash
  node structured_output_generation.js model_dir
  ```
  After running the command, an interactive dialog starts. You can enter a prompt and receive a structured output in response. The process is divided into two stages:

1. **Stage One:** The model generates a JSON schema indicating the number of items of each type the user requests. For example, if you prompt:  
   `Generate a JSON for 2 cars and 1 person with an Irish surname`  
   The model might output:  
   `{"person": 1, "car": 2, "transaction": 0}`  
   This internal JSON is used to determine how many items of each type to generate in the next stage. It is not shown to the user.

2. **Stage Two:** For each item type and count specified in the schema, the model is prompted to generate a JSON object. The original prompt is reused, but the schema guides the model to produce the correct structure. For the example above, the output might look like:
   ```
   > Generate a JSON for 2 cars and 1 person with an Irish surname
   output:
   {"name": "John Doe", "surname": "O'Reilly", "age": 30, "city": "Dublin"}
   {"model": "Toyota", "year": 2020, "engine": "hybrid"}
   {"model": "Ford", "year": 2019, "color": "red"}
   ```

**Note:**  
Structured output enforcement guarantees correct JSON formatting, but does not ensure the factual correctness or sensibility of the content. The model may generate implausible or nonsensical data, such as `{"name": "John", "age": 200000}` or `{"model": "AbrakaKadabra9999######4242"}`. These are valid JSONs but may not make sense. For best results, use the latest or fine-tuned models for this task to improve the quality and relevance of the generated output.


### 8. Tool Calling with Structural Tags Sample (`structural_tags_generation`)
- **Description:**
  Structural tags is a technique that allows to switch from regular sampling to structural output generation and back during the text generation.
  If during the sampling process the model produces a trigger string, it switches to structured mode and generates output according to a JSON schema defined by the tag. After that the model switches back to regular sampling mode.
  This is useful for generating function calls or other structured outputs that need to follow a specific format.

  This sample demonstrates how to use OpenVINO GenAI to generate structured tool calls from natural language prompts using structural tags. 
  The model is guided to output function calls in a specific format, enabling integration with external tools: 
  - Weather API 
  - Currency exchange APIs

  The system message instructs the model to call tools using a strict format:
  ```
  <function="function_name">
  {"argument1": "value1", ...}
  </function>
  ```
  The sample includes schemas for each tool, and the model is prompted to use them for tool calling. There are two model calls - with and without structural tags. 
  You can compare the results to see how the model generates structured outputs when using structural tags.
  If there is no prompt provided, the sample will use the default prompt: `"What is the weather in London today and in Paris yesterday, and how many pounds can I get for 100 euros?"`

- **Main Feature:** Structured tool call generation with LLM using schema enforcement with structural tags.
- **Run Command:**
  ```bash
  node structural_tags_generation.js model_dir [prompt]
  ```
  After running, the script will print the generated text output with and without structural tags, and display the parsed tool calls.

**Note:**  
This approach is useful for building LLM-powered agents that interact with external APIs or services in a controlled, structured way. 
For best results, use models fine-tuned for function calling and adapt structural tags according to the model function call template.
If the model does not generate trigger strings there will be no structural constraints during the generation. 
The sample is verified with `meta-llama/Llama-3.2-3B-Instruct` model. Other models may not produce the expected results or might require different system prompt.


### 9. Compound Grammar Generation Sample (`compound_grammar_generation`)
- **Description:**
  This sample demonstrates advanced structured output generation using compound grammars in OpenVINO GenAI.
  It showcases how to combine multiple grammar types - Regex, JSONSchema and EBNF - using Union and Concat operations to strictly control LLM output.
  It features multi-turn chat, switching grammar constraints between turns (e.g., "yes"/"no" answers and structured tool calls).
  Union operation allows the model to choose which grammar to use during generation. 
  In the sample it is used to combine two regex grammars for `"yes"` or `"no"` answer.
  Concat operation allows to start with one grammar and continue with another. 
  In the sample it used to create a `phi-4-mini-instruct` style tool calling answer - `functools[{tool_1_json}, ...]` - by combining regex and JSON schema grammars.

- **Main Features:**
  - Create grammar building blocks: Regex, JSONSchema, EBNF grammar
  - Combine grammars with Concat and Union operations
  - Multi-turn chat with grammar switching
  - Structured tool calling using zod schemas
- **Run Command:**
  ```bash
  node compound_grammar_generation.js model_dir
  ```
- **Notes:**
  This sample is ideal for scenarios requiring strict control over LLM outputs, such as building agents that interact with APIs or require validated structured responses. It showcases how to combine regex triggers and JSON schema enforcement for robust output generation.
  The sample is verified with `microsoft/Phi-4-mini-instruct` model. Other models may not produce the expected results or might require different system prompt.

#### Options
- `-m`, `--model`: Path to model and tokenizers base directory. [string] [required]
- `-p`, `--prompt`: The prompt to generate text. If without `-p` and `--pf`, the default prompt is `The Sky is blue because`. [string]
- `--prompt_file`, `--pf`: Read prompt from file. [string]
- `--num_warmup`, `--nw`: Number of warmup iterations. [number] [default: 1]
- `-n`, `--num_iter`: Number of iterations. [number] [default: 2]
- `--max_new_tokens`, `--mt`: Maximal number of new tokens. [number] [default: 20]
- `-d`, `--device`: Device to run the model on. [string] [default: "CPU"]

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
