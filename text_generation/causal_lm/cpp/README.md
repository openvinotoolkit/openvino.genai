# Text generation C++ samples that support most popular models like LLaMA 2

These examples showcase inference of text-generation Large Language Models (LLMs): `chatglm`, `LLaMA`, `Qwen` and other models with the same signature. The applications don't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. Loading `openvino_tokenizers` to `ov::Core` enables tokenization. Run `optimum-cli` to generate IRs for the samples. [group_beam_searcher.hpp](group_beam_searcher.hpp) implements the algorithm of the same name, which is used by `beam_search_causal_lm`. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot) which provides an example of LLM-powered Chatbot in Python.

## How it works

### Stateful LLM

A common LLM inference optimisation is introduction of past KV (key/value)-cache. This cache is represented by the corresponding inputs and outputs in a model implemented originally in DL framework (e.g. PyTorch models from HuggingFace). To optimize it further and simplify usage, the model is transformed to a stateful form. This transformation improves inference performance and decreases amount of allocated runtime memory in long running text generation scenarios. It is achieved by hiding inputs and outputs of the model that represent past KV-cache tensors and handling them inside the model in a more efficient way. Although the cache is still accessible with state API. It is opposed to stateless model approach requiring manipulating these inputs and outputs explicitly. An introduction to stateful models can be found in https://docs.openvino.ai/2023.3/openvino_docs_OV_UG_stateful_models_intro.html.

Hiding KV-cache introduces a peculiarity for beam search algorithm. Beam search suggests batched inference of multiple beams. The design described here so far would result in generating multiple independent sequences of tokens. Beam search algorithm, on the other hand, requires removing some of the ongoing beams and splitting other beams to multiple branches. Beam removal requires deleting corresponding KV-cache entry and beam splitting requires copying corresponding KV-cache values.

To provide the possibility to implement beam search without accessing model's internal state, a stateful LLM converted with `optimum-intel` or [llm_bench](../../../llm_bench/python/) introduces an additional 1-dimentional `beam_idx` input. `beam_idx` must contain indexes of elements in a batch which are intended to be selected and will evolve during the next beam search iteration. There's only one beam when the generation starts. That beam corresponds to the initial prompt. `beam_idx` must have values: `[0, 0]` to keep the initial beam and introduce its copy. The dynamic batch size enables to change the number of beams dynamically. `beam_idx` must have `[1]` as the value to remove zeroth sequence and keep the second beam only.

Assume there are two running beams. To proceed with generating both beams at the next iteration, `beam_idx` values must be `[0, 1]`, pointing to batch elements `0` and `1`. To drop the last beam and split the other beam in two, `beam_idx` must be set to `[0, 0]`. This results in utilizing only the part of KV cache corresponding to the zeroth element in the batch. The process of selecting proper entries in cache is called Cache Reorder.

![](beam_idx-fork.gif)
![](beam_idx-drop.gif)

The images below represent stateless and stateful LLM pipelines. The model has 4 inputs:
1. `input_ids` contains the next selected token
2. `attention_mask` is filled with `1`
3. `position_ids` encodes a position of currently generating token in the sequence
4. `beam_idx` selects beams

The model has 1 output `logits` describing the predicted distribution over the next tokens. And there's KV cache state.

![](stateless.jpg)
![](stateful.jpg)

### greedy_causal_lm

The program loads a tokenizer, a detokenizer and a model (`.xml` and `.bin`) to OpenVINO. A prompt is tokenized and passed to the model. The model greedily generates token by token until the special end of sequence (EOS) token is obtained. The predicted tokens are converted to chars and printed in a streaming fashion.

### beam_search_causal_lm

The program loads a tokenizer, a detokenizer and a model (`.xml` and `.bin`) to OpenVINO. A prompt is tokenized and passed to the model. The model predicts a distribution over the next tokens and group beam search samples from that distribution to explore possible sequesnses. The result is converted to chars and printed.

### speculative_decoding_lm

Speculative decoding (or [assisted-generation](https://huggingface.co/blog/assisted-generation#understanding-text-generation-latency) in HF terminology) is a recent technique, that allows to speed up token generation when an additional smaller draft model is used alonside with the main model.

Speculative decoding works the following way. The draft model predicts the next K tokens one by one in an autoregressive manner, while the main model validates these predictions and corrects them if necessary. We go through each predicted token, and if a difference is detected between the draft and main model, we stop and keep the last token predicted by the main model. Then the draft model gets the latest main prediction and again tries to predict the next K tokens, repeating the cycle.

This approach reduces the need for multiple infer requests to the main model, enhancing performance. For instance, in more predictable parts of text generation, the draft model can, in best-case scenarios, generate the next K tokens that exactly match the target. In tha caste the are validated in a single inference request to the main model (which is bigger, more accurate but slower) instead of running K subsequent requests. More details can be found in the original paper https://arxiv.org/pdf/2211.17192.pdf, https://arxiv.org/pdf/2302.01318.pdf

### prompt_lookup_decoding_lm

[Prompt Lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) is [assested-generation](https://huggingface.co/blog/assisted-generation#understanding-text-generation-latency) technique where the draft model is replaced with simple string matching the prompt to generate candidate token sequences. This method highly effective for input grounded generation (summarization, document QA, multi-turn chat, code editing), where there is high n-gram overlap between LLM input (prompt) and LLM output. This could be entity names, phrases, or code chunks that the LLM directly copies from the input while generating the output. Prompt lookup exploits this pattern to speed up autoregressive decoding in LLMs. This results in significant speedups with no effect on output quality.

> [!NOTE]
>Models should belong to the same family and have same tokenizers.

## Install OpenVINO

Install [OpenVINO Archives >= 2024.1](docs.openvino.ai/install). `master` and possibly the latest `releases/*` branch correspond to not yet released OpenVINO versions. https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/ can be used for these branches early testing. `<INSTALL_DIR>` below refers to the extraction location.

## Install `libtbb-dev` on Linux

> [!NOTE]
> `tbb` development files are installed with OpenVINO Archive on Windows and macOS.

```sh
sudo apt-get install libtbb-dev
```

## Build `greedy_causal_lm`, `beam_search_causal_lm` and `openvino_tokenizers`

### Linux/macOS

```sh
git submodule update --init
source <INSTALL_DIR>/setupvars.sh
cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/ && cmake --build ./build/ -j
```

### Windows

```bat
git submodule update --init
<INSTALL_DIR>\setupvars.bat
cmake -S .\ -B .\build\ && cmake --build .\build\ --config Release -j
```

### Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

#### Linux/macOS

```sh
source <INSTALL_DIR>/setupvars.sh
python3 -m pip install --upgrade-strategy eager -r requirements.txt
# Update openvino_tokenizers from the submodule
python3 -m pip install ./../../../thirdparty/openvino_tokenizers/[transformers]
optimum-cli export openvino --trust-remote-code --weight-format fp16 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
```

#### Windows

```bat
<INSTALL_DIR>\setupvars.bat
python -m pip install --upgrade-strategy eager -r requirements.txt
REM Update openvino_tokenizers from the submodule
python -m pip install .\..\..\..\thirdparty\openvino_tokenizers\[transformers]
optimum-cli export openvino --trust-remote-code --weight-format fp16 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
```

## Run

### Usage:
1. `greedy_causal_lm <MODEL_DIR> "<PROMPT>"`
2. `beam_search_causal_lm <MODEL_DIR> "<PROMPT>"`
3. `speculative_decoding_lm <DRAFT_MODEL_DIR> <MAIN_MODEL_DIR> "<PROMPT>"`
4. `prompt_lookup_decoding_lm <MODEL_DIR> "<PROMPT>"`

### Examples:

#### Linux/MacOS:
1. `./build/greedy_causal_lm ./TinyLlama-1.1B-Chat-v1.0/ "Why is the Sun yellow?"`
2. `./build/beam_search_causal_lm ./TinyLlama-1.1B-Chat-v1.0/ "Why is the Sun yellow?"`
3. `./build/speculative_decoding_lm ./TinyLlama-1.1B-Chat-v1.0/ ./Llama-2-7b-chat-hf/ "Why is the Sun yellow?"`
4. `./build/prompt_lookup_decoding_lm ./TinyLlama-1.1B-Chat-v1.0/ "Why is the Sun yellow?"`

#### Windows:
1. `.\build\Release\greedy_causal_lm .\TinyLlama-1.1B-Chat-v1.0\ "Why is the Sun yellow?"`
2. `.\build\Release\beam_search_causal_lm .\TinyLlama-1.1B-Chat-v1.0\ "Why is the Sun yellow?"`
3. `.\build\Release\speculative_decoding_lm .\TinyLlama-1.1B-Chat-v1.0\ .\Llama-2-7b-chat-hf\ "Why is the Sun yellow?"`
4. `.\build\Release\prompt_lookup_decoding_lm .\TinyLlama-1.1B-Chat-v1.0\ "Why is the Sun yellow?"`

To enable Unicode characters for Windows cmd open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.

Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM. For example, the model meta-llama/Llama-2-13b-chat-hf can benefit from being run on a dGPU. Modify the source code to change the device for inference to the GPU.

## Supported models

1. chatglm
   1. https://huggingface.co/THUDM/chatglm2-6b - refer to
   [chatglm2-6b - AttributeError: can't set attribute](../../../llm_bench/python/doc/NOTES.md#chatglm2-6b---attributeerror-cant-set-attribute)
   in case of `AttributeError`
   2. https://huggingface.co/THUDM/chatglm3-6b
2. LLaMA 2 (requires access request submission on its Hugging Face page to be downloaded)
   1. https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
   2. https://huggingface.co/meta-llama/Llama-2-13b-hf
   3. https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
   4. https://huggingface.co/meta-llama/Llama-2-7b-hf
   5. https://huggingface.co/meta-llama/Llama-2-70b-chat-hf
   6. https://huggingface.co/meta-llama/Llama-2-70b-hf
3. [Llama2-7b-WhoIsHarryPotter](https://huggingface.co/microsoft/Llama2-7b-WhoIsHarryPotter)
4. OpenLLaMA
   1. https://huggingface.co/openlm-research/open_llama_13b
   2. https://huggingface.co/openlm-research/open_llama_3b
   3. https://huggingface.co/openlm-research/open_llama_3b_v2
   4. https://huggingface.co/openlm-research/open_llama_7b
   5. https://huggingface.co/openlm-research/open_llama_7b_v2
5. [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
6. Qwen
   1. https://huggingface.co/Qwen/Qwen-7B-Chat
   2. https://huggingface.co/Qwen/Qwen-7B-Chat-Int4 - refer to
   3. https://huggingface.co/Qwen/Qwen1.5-7B-Chat
   4. https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GPTQ-Int4
   [Qwen-7B-Chat-Int4 - Torch not compiled with CUDA enabled](../../../llm_bench/python/doc/NOTES.md#qwen-7b-chat-int4---torch-not-compiled-with-cuda-enabled)
   in case of `AssertionError`
7. Dolly
   1. https://huggingface.co/databricks/dolly-v2-3b
8. Phi
   1. https://huggingface.co/microsoft/phi-2
   2. https://huggingface.co/microsoft/phi-1_5
9. [notus-7b-v1](https://huggingface.co/argilla/notus-7b-v1)
10. [zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
11. [redpajama-3b-chat](https://huggingface.co/ikala/redpajama-3b-chat)
12. [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
13. [Gemma-2B-it](https://huggingface.co/google/gemma-2b-it)

This pipeline can work with other similar topologies produced by `optimum-intel` with the same model signature.

Some models may require access request submission on their Hugging Face page to be downloaded.

If https://huggingface.co/ is down, the conversion step won't be able to download the models.
