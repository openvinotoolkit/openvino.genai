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
optimum-cli export openvino --model <model> <output_folder>
```
If a converted model in OpenVINO IR format is already available in the collection of [OpenVINO optimized LLMs](https://huggingface.co/collections/OpenVINO/llm-6687aaa2abca3bbcec71a9bd) on Hugging Face, it can be downloaded directly via huggingface-cli.
```sh
pip install huggingface-hub
huggingface-cli download <model> --local-dir <output_folder>
```

## Sample Descriptions
### Common information
Follow [Get Started with Samples](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/get-started-demos.html) to get common information about OpenVINO samples.
Follow [build instruction](../../../src/docs/BUILD.md) to build GenAI samples

GPUs usually provide better performance compared to CPUs. Modify the source code to change the device for inference to the GPU.

See https://github.com/openvinotoolkit/openvino.genai/blob/master/SUPPORTED_MODELS.md for the list of supported models.

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

Recommended models: meta-llama/Llama-2-13b-hf as main model and TinyLlama/TinyLlama-1.1B-Chat-v1.0 as draft model, etc
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
This sample script demonstrates how to benchmark an LLMs in OpenVINO GenAI. The script includes functionality for warm-up iterations, generating text, and calculating various performance metrics.

For more information how performance metrics are calculated please follow [performance-metrics tutorial](../../../src/README.md#performance-metrics).
- **Main Feature:** Benchmark model via GenAI
- **Run Command:**
  ```bash
  python benchmark_genai.py [-m MODEL] [-p PROMPT] [-nw NUM_WARMUP] [-n NUM_ITER] [-mt MAX_NEW_TOKENS] [-d DEVICE]
  ```
  #### Options
- `-m, --model`: Path to the model and tokenizers base directory.
- `-p, --prompt` (default: `"The Sky is blue because"`): The prompt to generate text.
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
This sample demonstrates how to use OpenVINO GenAI to generate structured outputs, such as JSON or other formats, from text prompts.

- **Run Command:**
  ```bash
  python structured_output_generation.py model_dir
  ```
After that a dialog will be started, where you can enter a prompt and get a structured output in response. Generation is separated into stages.
At the first stage, the model generates a JSON for number of each items user want. And at the second stage, `generate` is called for each item to generate a JSON for each object car, person, or transaction. This is doen because not all models are able to generate a combplex JSON, with variadic number of elements, in one shot, expecially if model is small, and 
not fine tuned for this task. But separation of tasks allows to use smaller models, and still get a good quality of generated JSON.

On the first stage models is prompted to generate a JSON schema for the number of items the user wants. For example, if the user wants to generate a `JSON for 2 cars, and 1 one person with Irish surname` the output JSON 
will be `{'person': 1, 'car': 2, 'transaction': 0}`. This means generate will be called 2 times with json schemes for cars, and 1 time for a person with josn shema for a person. This is an internal json, it's not shown to the user, but it is used to determine how many items of each type the model should generate in the next stage.

On the second stage, model still receives the first prompt, but json schema is applied for the exact item type, so the model knows what to generate. With the prompt above it might look like this:
```
> Generate a JSON for 2 cars, and 1 one person with Irish surname
output:
{"name": "John Doe", "surname: "O'Reilly", "age": 30, "city: "Dublin"}
{"model": "Toyota", "year": 2020, "engine": "hybrid"}
{"model": "Ford", "year": 2019, "color": "red"}
```

Please not that strctured output enforced guaranties the correct json format, but does not guarantee the correctness of the content. The model may generate incorrect or nonsensical data, e.g. `{"name": "John",... , 'age' 200000}`, or a car with `{"model": "AbrakaKadabra9999######4242",...}` these are nonsencical values but they are still valid JSON.

Please use the latest or fine tunes models for this taks in order to get the sensible output.

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
