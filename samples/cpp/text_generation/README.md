# OpenVINO GenAI Text Generation Samples

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
optimim-cli export openvino --model <model> <output_folder>
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

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/) for more details.

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
  ./chat_sample <MODEL_DIR>
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
  ./greedy_causal_lm <MODEL_DIR> "<PROMPT>"
  ```

### 3. Beam Search Causal LM (`beam_search_causal_lm`)
- **Description:**
Uses beam search for more coherent text generation.
Here is a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-question-answering) that provides an example of LLM-powered text generation in Python.
Recommended models: meta-llama/Llama-2-7b-hf, etc
- **Main Feature:** Improves text quality with beam search.
- **Run Command:**
  ```bash
  ./beam_search_causal_lm <MODEL_DIR> "<PROMPT 1>" ["<PROMPT 2>" ...]
  ```

### 4. Multinomial Causal LM (`multinomial_causal_lm`)
- **Description:** Text generation with multinomial sampling for diversity.
Recommended models: meta-llama/Llama-2-7b-hf, etc
- **Main Feature:** Introduces randomness for creative outputs.
- **Run Command:**
  ```bash
  ./multinomial_causal_lm <MODEL_DIR> "<PROMPT>"
  ```

### 5. Prompt Lookup Decoding LM (`prompt_lookup_decoding_lm`)
- **Description:** 
[Prompt Lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) is [assested-generation](https://huggingface.co/blog/assisted-generation#understanding-text-generation-latency) technique where the draft model is replaced with simple string matching the prompt to generate candidate token sequences. This method highly effective for input grounded generation (summarization, document QA, multi-turn chat, code editing), where there is high n-gram overlap between LLM input (prompt) and LLM output. This could be entity names, phrases, or code chunks that the LLM directly copies from the input while generating the output. Prompt lookup exploits this pattern to speed up autoregressive decoding in LLMs. This results in significant speedups with no effect on output quality.
Recommended models: meta-llama/Llama-2-7b-hf, etc
- **Main Feature:** Specialized prompt-based inference.
- **Run Command:**
  ```bash
  ./prompt_lookup_decoding_lm <MODEL_DIR> "<PROMPT>"
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
  ./speculative_decoding_lm <MODEL_DIR> <DRAFT_MODEL_DIR> "<PROMPT>"
  ```

### 7. LoRA Greedy Causal LM (`lora_greedy_causal_lm`)
- **Description:**
This sample demonstrates greedy decoding using Low-Rank Adaptation (LoRA) fine-tuned causal language models. LoRA enables efficient fine-tuning, reducing resource requirements for adapting large models to specific tasks.
- **Main Feature:** Lightweight fine-tuning with LoRA for efficient text generation
- **Run Command:**
  ```bash
  ./lora_greedy_causal_lm <MODEL_DIR> <ADAPTER_SAFETENSORS_FILE> "<PROMPT>"
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
The following code snippet demonstrates how to load the model from the memory buffer:
```cpp
auto [model_str, weights_tensor] = decrypt_model(models_path + "/openvino_model.xml", models_path + "/openvino_model.bin");
ov::genai::Tokenizer tokenizer(models_path);
ov::genai::LLMPipeline pipe(model_str, weights_tensor, tokenizer, device);
```
For the sake of brevity the code above does not include Tokenizer decryption. For more details look to encrypted_model_causal_lm sample.
The sample also demonstrates how to enable user defined encryption for plugin cache.
- **Main Feature:** Read model directly from memory buffer
- **Run Command:**
  ```bash
  ./encrypted_model_causal_lm <MODEL_DIR> "<PROMPT>"
  ```

### 9. LLMs benchmarking sample (`benchmark_genai`)
- **Description:** 
This sample script demonstrates how to benchmark an LLMs in OpenVINO GenAI. The script includes functionality for warm-up iterations, generating text, and calculating various performance metrics.

For more information how performance metrics are calculated please follow [performance-metrics tutorial](../../../src/README.md#performance-metrics).
- **Main Feature:** Benchmark model via GenAI
- **Run Command:**
  ```bash
  ./benchmark_genai [OPTIONS]
  ```
  #### Options
- `-m, --model`: Path to the model and tokenizers base directory.
- `-p, --prompt` (default: ''): The prompt to generate text. If without `-p` and `--pf`, the default prompt is `"The Sky is blue because"`
- `--pf, --prompt_file` Read prompt from file.
- `--nw, --num_warmup` (default: `1`): Number of warmup iterations.
- `--mt, --max_new_tokens` (default: `20`): Maximal number of new tokens.
- `-n, --num_iter` (default: `3`): Number of iterations.
- `-d, --device` (default: `"CPU"`): Device to run the model on.

### 10. Structured Output Sample (`structured_output_sample`)
- **Description:**  
This sample demonstrates how to use OpenVINO GenAI to generate structured outputs, such as JSON, from text prompts. It showcases step-by-step reasoning, allowing a language model to break down tasks (e.g., solving equations) and present each step in a structured format.

The sample uses the following JSON schema for structured output:
```json
{
  "steps": [
    ...
    {"explanation": "Moving the -30 term to the right", "output": "2*x = -30"},
    {"explanation": "Finding the value of x.", "output": "x = -30/2"}
    ...
  ],
  "final_answer": "x = -15"
}
```
**Schema Details:**
- Each reasoning step is an object with `explanation` and `output` fields.
- The `steps` array lists all steps in order.
- The `final_answer` field provides the final solution.
- The schema is defined in the sample source code and can be customized as needed.
 - JSON schema for such format is defined in the source code of the sample, and can be modified to fit your needs.

Recommended models: `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-8B-Instruct`

- **Run Command:**
  ```bash
  structured_output_generation <MODEL_DIR>
  ```
  After running the command, an interactive dialog starts. You can prompt the model to solve equations step by step. For example:

1. **Step-by-step reasoning:**  
   If you prompt:  
   `Solve the equation 8x + 7 = -23 step by step`  
   The model might output:
   ```json
   {
     "steps": [
       {"explanation": "Rearranging the equation to isolate x.", "output": "8x + 7 = -23"},
       {"explanation": "Subtracting 7 from both sides.", "output": "8x + 7 - 7 = -23 - 7"},
       {"explanation": "Simplifying the left side.", "output": "8x = -30"},
       {"explanation": "Dividing both sides by 8.", "output": "8x / 8 = -30 / 8"},
       {"explanation": "Simplifying the right side.", "output": "x = -30 / 8"},
       {"explanation": "Finding the value of x.", "output": "x = -15/4"}
     ],
     "final_answer": "x = -15/4"
   }
   ```

**Note:**  
Structured output enforcement ensures valid JSON formatting, but does not guarantee factual accuracy or meaningfulness. The model may generate plausible-looking JSON with incorrect or nonsensical data (e.g., `{"explanation": "John", "output": 200000}` or `{"final_answer": "AbrakaKadabra9999######4242"}`). For best results, use the latest or fine-tuned models to improve output quality and relevance.

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
