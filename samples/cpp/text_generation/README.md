
# OpenVINO AI Text Generation Samples

These samples showcase the use of OpenVINO's inference capabilities for text generation tasks, including different decoding strategies such as beam search, multinomial sampling, and speculative decoding. Each sample has a specific focus and demonstrates a unique aspect of text generation.
The applications don't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU.
There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-chatbot) that provides an example of LLM-powered text generation in Python.

## Table of Contents
1. [Download and Convert the Model and Tokenizers](#download-and-convert-the-model-and-tokenizers)
2. [Running the Samples](#running-the-samples)
3. [Using encrypted models](#using-encrypted-models)
4. [Sample Descriptions](#sample-descriptions)
5. [Troubleshooting](#troubleshooting)
6. [Support and Contribution](#support-and-contribution)

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

It's not required to install [../../export-requirements.txt](../../export requirements.txt) for deployment if the model has already been exported.

```sh
pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --trust-remote-code --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
```

## Running the Samples

Follow [Get Started with Samples](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/get-started-demos.html) to run a specific sample.

`greedy_causal_lm TinyLlama-1.1B-Chat-v1.0 "Why is the Sun yellow?"`

Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM. For example, the model meta-llama/Llama-2-13b-chat-hf can benefit from being run on a dGPU. Modify the source code to change the device for inference to the GPU.

See https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md#supported-models for the list of supported models.


## Sample Descriptions

### 1. Greedy Causal LM (`greedy_causal_lm`)
- **Description:** Basic text generation using a causal language model.
- **Main Feature:** Demonstrates simple text continuation.
- **Run Command:**
  ```bash
  ./greedy_causal_lm <model_path> <prompt>
  ```

### 2. Beam Search Causal LM (`beam_search_causal_lm`)
- **Description:** Uses beam search for more coherent text generation.
- **Main Feature:** Improves text quality with beam search.
- **Run Command:**
  ```bash
  ./beam_search_causal_lm <model_path> <prompt>
  ```

### 3. Chat Sample (`chat_sample`)
- **Description:** Interactive chat interface powered by OpenVINO.
- **Main Feature:** Real-time chat-like text generation.
- **Run Command:**
  ```bash
  ./chat_sample <model_path>
  ```
#### Missing chat template
If you encounter an exception indicating a missing "chat template" when launching the `ov::genai::LLMPipeline` in chat mode, it likely means the model was not tuned for chat functionality. To work this around, manually add the chat template to tokenizer_config.json of your model.
The following template can be used as a default, but it may not work properly with every model:
```
"chat_template": "{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n<|im_start|>assistant\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|im_end|>\n'}}{% endif %}{% endfor %}",
```


### 4. Multinomial Causal LM (`multinomial_causal_lm`)
- **Description:** Text generation with multinomial sampling for diversity.
- **Main Feature:** Introduces randomness for creative outputs.
- **Run Command:**
  ```bash
  ./multinomial_causal_lm <model_path> <prompt>
  ```

### 5. Prompt Lookup Decoding LM (`prompt_lookup_decoding_lm`)
- **Description:** 
[Prompt Lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) is [assested-generation](https://huggingface.co/blog/assisted-generation#understanding-text-generation-latency) technique where the draft model is replaced with simple string matching the prompt to generate candidate token sequences. This method highly effective for input grounded generation (summarization, document QA, multi-turn chat, code editing), where there is high n-gram overlap between LLM input (prompt) and LLM output. This could be entity names, phrases, or code chunks that the LLM directly copies from the input while generating the output. Prompt lookup exploits this pattern to speed up autoregressive decoding in LLMs. This results in significant speedups with no effect on output quality.
- **Main Feature:** Specialized prompt-based inference.
- **Run Command:**
  ```bash
  ./prompt_lookup_decoding_lm <model_path> <prompt>
  ```

### 6. Speculative Decoding LM (`speculative_decoding_lm`)
- **Description:** 
Speculative decoding (or [assisted-generation](https://huggingface.co/blog/assisted-generation#understanding-text-generation-latency) in HF terminology) is a recent technique, that allows to speed up token generation when an additional smaller draft model is used alongside with the main model.

Speculative decoding works the following way. The draft model predicts the next K tokens one by one in an autoregressive manner, while the main model validates these predictions and corrects them if necessary. We go through each predicted token, and if a difference is detected between the draft and main model, we stop and keep the last token predicted by the main model. Then the draft model gets the latest main prediction and again tries to predict the next K tokens, repeating the cycle.

This approach reduces the need for multiple infer requests to the main model, enhancing performance. For instance, in more predictable parts of text generation, the draft model can, in best-case scenarios, generate the next K tokens that exactly match the target. In that case they are validated in a single inference request to the main model (which is bigger, more accurate but slower) instead of running K subsequent requests. More details can be found in the original paper https://arxiv.org/pdf/2211.17192.pdf, https://arxiv.org/pdf/2302.01318.pdf
- **Main Feature:** Reduces latency while generating high-quality text.
- **Run Command:**
  ```bash
  ./speculative_decoding_lm <main_model_path> <draft_model_path> <prompt>
  ```

### 7. Encrypted Model Causal LM (`encrypted_model_causal_lm`)
- **Description:** 
LLMPipeline and Tokenizer objects can be initialized directly from the memory buffer, e.g. when user stores only encrypted files and decrypts them on-the-fly. 
The following code snippet demonstrates how to load the model from the memory buffer:
```cpp
auto [model_str, weights_tensor] = decrypt_model(models_path + "/openvino_model.xml", models_path + "/openvino_model.bin");
ov::genai::Tokenizer tokenizer(models_path);
ov::genai::LLMPipeline pipe(model_str, weights_tensor, tokenizer, device);
```
For the sake of brevity the code above does not include Tokenizer decryption. For more details look to encrypted_model_causal_lm sample.
- **Main Feature:** Read model directly from memory buffer
- **Run Command:**
  ```bash
  ./encrypted_model_causal_lm <model_path> <prompt>
  ```

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
