# OpenVINO GenAI Text Generation Python Samples

These samples showcase the use of OpenVINO's inference capabilities for text generation tasks, including different decoding strategies such as beam search, multinomial sampling, and speculative decoding. Each sample has a specific focus and demonstrates a unique aspect of text generation.
The applications don't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU.
There are also Jupyter notebooks for some samples. You can find links to them in the appropriate sample descriptions.

## Table of Contents
1. [Download and Convert the Model and Tokenizers](#download-and-convert-the-model-and-tokenizers)
2. [Sample Descriptions](#sample-descriptions)
3. [Troubleshooting](#troubleshooting)
4. [Support and Contribution](#support-and-contribution)

## Download and convert the model and tokenizers
The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.
Install [../../export-requirements.txt](../../export-requirements.txt) if model conversion is required.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
```

Then, run the export with Optimum CLI:

```sh
optimum-cli export openvino --model <model> <output_folder>
```

Alternatively, do it in Python code (e.g. TinyLlama_v1.1). If NNCF is installed, the model will be compressed to INT8 automatically.

```python
from optimum.exporters.openvino.convert import export_tokenizer
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

output_dir = "chat_model"

model = OVModelForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1", export=True)
model.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
tokenizer.save_pretrained(output_dir)
export_tokenizer(tokenizer, output_dir)
```
[//]: # "tokenizer.save_pretrained(output_dir) is required above to mitigate runtime errors"

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
Follow [Get Started with Samples](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/get-started-demos.html) to get common information about OpenVINO samples.
Follow [build instruction](../../../src/docs/BUILD.md) to build GenAI samples

GPUs usually provide better performance compared to CPUs. Modify the source code to change the device for inference to the GPU.

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#large-language-models-llms) for more details.

Install [../../deployment-requirements.txt](../../deployment-requirements.txt) to run samples
```sh
pip install --upgrade-strategy eager -r ../../deployment-requirements.txt
```

### 1. Chat Sample (`chat_sample`)
- **Description:**
Interactive chat interface powered by OpenVINO.
Here is a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-chatbot) that provides an example of LLM-powered text generation in Python.
Recommended models: meta-llama/Llama-2-7b-chat-hf, TinyLlama/TinyLlama-1.1B-Chat-v1.0, etc
- **Main Feature:** Real-time chat-like text generation.
- **Run Command:**
  ```bash
  python chat_sample.py model_dir
  ```
#### Missing chat template
If you encounter an exception indicating a missing "chat template" when launching the `ov::genai::LLMPipeline` in chat mode, it likely means the model was not tuned for chat functionality. To work this around, manually add the chat template to tokenizer_config.json of your model or update it using call `pipe.get_tokenizer().set_chat_template(new_chat_template)`.
The following template can be used as a default, but it may not work properly with every model:
```
"chat_template": "{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n<|im_start|>assistant\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|im_end|>\n'}}{% endif %}{% endfor %}",
```

#### NPU support

NPU device is supported with some limitations. See [NPU inference of
LLMs](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html) documentation. In particular:

- Models must be exported with symmetric INT4 quantization (`optimum-cli export openvino --weight-format int4 --sym --model <model> <output_folder>`).
  For models with more than 4B parameters, channel wise quantization should be used (`--group-size -1`).
- Beam search and parallel sampling are not supported.
- Use OpenVINO 2025.0 or later (installed by deployment-requirements.txt, see "Common information" section), and the latest NPU driver.


### 2. Greedy Causal LM (`greedy_causal_lm`)
- **Description:**
Basic text generation using a causal language model.
Here is a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-question-answering) that provides an example of LLM-powered text generation in Python.
Recommended models: meta-llama/Llama-2-7b-hf, etc
- **Main Feature:** Demonstrates simple text continuation.
- **Run Command:**
  ```bash
  python greedy_causal_lm.py [-h] model_dir prompt
  ```

### 3. Beam Search Causal LM (`beam_search_causal_lm`)
- **Description:**
Uses beam search for more coherent text generation.
Here is a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-question-answering) that provides an example of LLM-powered text generation in Python.
Recommended models: meta-llama/Llama-2-7b-hf, etc
- **Main Feature:** Improves text quality with beam search.
- **Run Command:**
  ```bash
  python beam_search_causal_lm.py model_dir prompt [prompts ...]
  ```

### 4. Multinomial Causal LM (`multinomial_causal_lm`)
- **Description:** Text generation with multinomial sampling for diversity.
Recommended models: meta-llama/Llama-2-7b-hf, etc
- **Main Feature:** Introduces randomness for creative outputs.
- **Run Command:**
  ```bash
  python multinomial_causal_lm.py model_dir prompt
  ```

### 5. Prompt Lookup Decoding LM (`prompt_lookup_decoding_lm`)
- **Description:** 
[Prompt Lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) is [assested-generation](https://huggingface.co/blog/assisted-generation#understanding-text-generation-latency) technique where the draft model is replaced with simple string matching the prompt to generate candidate token sequences. This method highly effective for input grounded generation (summarization, document QA, multi-turn chat, code editing), where there is high n-gram overlap between LLM input (prompt) and LLM output. This could be entity names, phrases, or code chunks that the LLM directly copies from the input while generating the output. Prompt lookup exploits this pattern to speed up autoregressive decoding in LLMs. This results in significant speedups with no effect on output quality.
Recommended models: meta-llama/Llama-2-7b-hf, etc
- **Main Feature:** Specialized prompt-based inference.
- **Run Command:**
  ```bash
  python prompt_lookup_decoding_lm.py model_dir prompt
  ```

### 6. Speculative Decoding LM (`speculative_decoding_lm`)
- **Description:** 
Speculative decoding (or [assisted-generation](https://huggingface.co/blog/assisted-generation#understanding-text-generation-latency) in HF terminology) is a recent technique, that allows to speed up token generation when an additional smaller draft model is used alongside with the main model.

Speculative decoding works the following way. The draft model predicts the next K tokens one by one in an autoregressive manner, while the main model validates these predictions and corrects them if necessary. We go through each predicted token, and if a difference is detected between the draft and main model, we stop and keep the last token predicted by the main model. Then the draft model gets the latest main prediction and again tries to predict the next K tokens, repeating the cycle.

This approach reduces the need for multiple infer requests to the main model, enhancing performance. For instance, in more predictable parts of text generation, the draft model can, in best-case scenarios, generate the next K tokens that exactly match the target. In that case they are validated in a single inference request to the main model (which is bigger, more accurate but slower) instead of running K subsequent requests. More details can be found in the original paper https://arxiv.org/pdf/2211.17192.pdf, https://arxiv.org/pdf/2302.01318.pdf

Here is a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/speculative-sampling) that provides an example of LLM-powered text generation in Python.

Recommended models: meta-llama/Llama-2-13b-hf as main model and TinyLlama/TinyLlama-1.1B-Chat-v1.0 as draft model. Note that GGUF models are not supported as draft models.
- **Main Feature:** Reduces latency while generating high-quality text.
- **Run Command:**
  ```bash
  python speculative_decoding_lm.py model_dir draft_model_dir prompt
  ```

### 7. LoRA Greedy Causal LM (`lora_greedy_causal_lm`)
- **Description:**
This sample demonstrates greedy decoding using Low-Rank Adaptation (LoRA) fine-tuned causal language models. LoRA enables efficient fine-tuning, reducing resource requirements for adapting large models to specific tasks.
- **Main Feature:** Lightweight fine-tuning with LoRA for efficient text generation
- **Run Command:**
  ```bash
  python lora_greedy_causal_lm.py model_dir adapter_safetensors_file prompt
  ```

> [!NOTE]
> ### LoRA `alpha` interpretation in OpenVINO GenAI
> The OpenVINO GenAI implementation merges the traditional LoRA parameters into a **single effective scaling factor** used during inference.
>
> In this context, the `alpha` value already includes:
> - normalization by LoRA rank (`alpha / rank`)
> - any user-defined scaling factor (`weight`)
>
> This means `alpha` in GenAI should be treated as the **final scaling weight** applied to the LoRA update — not the raw `alpha` parameter from training.

### 8. Encrypted Model Causal LM (`encrypted_model_causal_lm`)
- **Description:** 
LLMPipeline and Tokenizer objects can be initialized directly from the memory buffer, e.g. when user stores only encrypted files and decrypts them on-the-fly. 
- **Main Feature:** Read model directly from memory buffer
- **Run Command:**
  ```bash
  python encrypted_model_causal_lm.py model_dir prompt
  ```

### 9. LLMs benchmarking sample (`benchmark_genai`)
- **Description:** 
This sample script demonstrates how to benchmark LLMs in OpenVINO GenAI. The script includes functionality for warm-up iterations, generating text, and calculating various performance metrics.

For more information how performance metrics are calculated, please follow the [performance-metrics tutorial](../../../src/README.md#performance-metrics).
- **Main Feature:** Benchmark model via GenAI
- **Run Command:**
  ```bash
  python benchmark_genai.py [-m MODEL] [-p PROMPT] [-nw NUM_WARMUP] [-n NUM_ITER] [-mt MAX_NEW_TOKENS] [-d DEVICE]
  ```
  #### Options
- `-m, --model`: Path to the model and tokenizers base directory.
- `-p, --prompt` (default: `None`): The prompt to generate text. If without `-p` and `-pf`, the default prompt is `"The Sky is blue because"`
- `-pf, --prompt_file` Read prompt from file.
- `-nw, --num_warmup` (default: `1`): Number of warmup iterations.
- `-mt, --max_new_tokens` (default: `20`): Maximal number of new tokens.
- `-n, --num_iter` (default: `3`): Number of iterations.
- `-d, --device` (default: `"CPU"`): Device to run the model on.

### 10. LLM ReAct Agent Sample (`react_sample`)
- **Description:**
Interactive ReAct Agent powered by OpenVINO.
Here is a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-native-agent-react) that provides an example of LLM-powered reasoning engine to execute an action in Python.
Recommended models: Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct
- **Main Feature:** Real-time reasoning-action from user's input.
- **Run Command:**
  ```bash
  python react_sample.py model_dir
  ```


### 11. Structured Output Sample (`structured_output_sample`)
- **Description:**
This sample demonstrates how to use OpenVINO GenAI to generate structured outputs such as JSON from text prompts. This sample implementation is split into multiple "generate" calls to mitigate generating complex, variadic JSON structures in a single pass. This is done because not all models are able to generate a complex JSON, with a variadic number of elements in one shot, especially if the model is small and not fine-tuned for this task. By separating the task into two stages, it becomes possible to use smaller models and still achieve generated JSON good quality.

Recommended models: meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-8B-Instruct
- **Run Command:**
  ```bash
  python structured_output_generation.py model_dir
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


### 12. Tool Calling with Structural Tags Sample (`structural_tags_generation`)
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
  python structural_tags_generation.py model_dir [--prompt "Your prompt here"]
  ```
  After running, the script will print the generated text output with and without structural tags, and display the parsed tool calls.

**Note:**  
This approach is useful for building LLM-powered agents that interact with external APIs or services in a controlled, structured way. 
For best results, use models fine-tuned for function calling and adapt structural tags according to the model function call template.
If the model does not generate trigger strings there will be no structural constraints during the generation. 
The sample is verified with `meta-llama/Llama-3.2-3B-Instruct` model. Other models may not produce the expected results or might require different system prompt.


### 13. Compound Grammar Generation Sample (`compound_grammar_generation`)
- **Description:**
  This sample demonstrates advanced structured output generation using compound grammars in OpenVINO GenAI.
  It showcases how to combine multiple grammar types - Regex, JSONSchema and EBNF - using Union (`|`) and Concat (`+`) operations to strictly control LLM output.
  It features multi-turn chat, switching grammar constraints between turns (e.g., "yes"/"no" answers and structured tool calls).
  Union (`|`) operation allows the model to choose which grammar to use during generation. 
  In the sample it is used to combine two regex grammars for `"yes"` or `"no"` answer.
  Concat (`+`) operation allows to start with one grammar and continue with another. 
  In the sample it used to create a `phi-4-mini-instruct` style tool calling answer - `functools[{tool_1_json}, ...]` - by combining regex and JSON schema grammars.

- **Main Features:**
  - Create grammar building blocks: Regex, JSONSchema, EBNF grammar
  - Combine grammars with Concat (`+`) and Union (`|`) operations
  - Multi-turn chat with grammar switching
  - Structured tool calling using Pydantic schemas
- **Run Command:**
  ```bash
  python compound_grammar_generation.py model_dir
  ```
- **Notes:**
  This sample is ideal for scenarios requiring strict control over LLM outputs, such as building agents that interact with APIs or require validated structured responses. It showcases how to combine regex triggers and JSON schema enforcement for robust output generation.
  The sample is verified with `microsoft/Phi-4-mini-instruct` model. Other models may not produce the expected results or might require different system prompt.


## Troubleshooting

### Unicode characters encoding error on Windows

Example error:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u25aa' in position 0: character maps to <undefined>
```

If you encounter the error described in the example when sample is printing output to the Windows console, it is likely due to the default Windows encoding not supporting certain Unicode characters. To resolve this:
1. Enable Unicode characters for Windows cmd - open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.
2. Enable UTF-8 mode by setting environment variable `PYTHONIOENCODING="utf8"`.

## Support and Contribution
- For troubleshooting, consult the [OpenVINO documentation](https://docs.openvino.ai).
- To report issues or contribute, visit the [GitHub repository](https://github.com/openvinotoolkit/openvino.genai).
