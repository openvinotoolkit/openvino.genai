# OpenVINO™ GenAI Library

## Install OpenVINO™ GenAI

The OpenVINO GenAI flavor is available for installation via Archive and PyPI distributions:
- [Archive Installation](./docs/INSTALL_ARCHIVE.md)
- [PyPI Installation](./docs/INSTALL_PYPI.md)

To build OpenVINO™ GenAI library from source, refer to the [Build Instructions](./docs/BUILD.md).

## Usage

For Python and C++ usage examples, refer to the [Generate API Usage Guide](./docs/USAGE.md).

## How it works

For information on how OpenVINO™ GenAI works, refer to the [How It Works Section](./docs/HOW_IT_WORKS.md).

## Supported models

For a list of supported models, refer to the [Supported Models Section](./docs/SUPPORTED_MODELS.md).

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


