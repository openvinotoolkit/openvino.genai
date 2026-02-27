<div align="center">

![OpenVINO GenAI](/site/static/img/openvino-genai-logo-gradient.svg)

[<b>Getting Started</b>](#getting-started) •
[<b>AI Scenarios</b>](#ai-scenarios) •
[<b>Optimization Methods</b>](#optimization-methods) •
[<b>Documentation</b>](https://openvinotoolkit.github.io/openvino.genai/)

[![GitHub Release](https://img.shields.io/github/v/release/openvinotoolkit/openvino.genai?color=green)](https://github.com/openvinotoolkit/openvino.genai/releases)
[![PyPI Downloads](https://static.pepy.tech/badge/openvino.genai)](https://pypi.org/project/openvino.genai/)
![Python](https://img.shields.io/badge/python-3.10+-green)
![OS](https://img.shields.io/badge/OS-Linux_|_Windows_|_MacOS-blue)

![](/site/static/img/openvino-genai-workflow.svg)

</div>

OpenVINO™ GenAI is a library of the most popular Generative AI model pipelines, optimized execution methods, and samples that run on top of highly performant [OpenVINO Runtime](https://github.com/openvinotoolkit/openvino).

This library is friendly to PC and laptop execution, and optimized for resource consumption. It requires no external dependencies to run generative models as it already includes all the core functionality (e.g. tokenization via [`openvino-tokenizers`](https://github.com/openvinotoolkit/openvino_tokenizers)).

![Text generation using LLaMa 3.2 model running on Intel ARC770 dGPU](./samples/generation.gif)

<a id="getting-started"></a>

## Getting Started

* [Introduction to OpenVINO™ GenAI](https://openvinotoolkit.github.io/openvino.genai/docs/getting-started/introduction)
* [Install OpenVINO™ GenAI](https://openvinotoolkit.github.io/openvino.genai/docs/getting-started/installation)
* [Build OpenVINO™ GenAI](/src/docs/BUILD.md)
* [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/)
* [Model Preparation Guide](https://openvinotoolkit.github.io/openvino.genai/docs/category/model-preparation)

Explore blogs to setup your first hands-on experience with OpenVINO GenAI:

* [How to Build OpenVINO™ GenAI APP in C++](https://medium.com/openvino-toolkit/how-to-build-openvino-genai-app-in-c-32dcbe42fa67)
* [How to run Llama 3.2 locally with OpenVINO™](https://medium.com/openvino-toolkit/how-to-run-llama-3-2-locally-with-openvino-60a0f3674549)

<a id="ai-scenarios"></a>

## Quick Start

1. Install OpenVINO GenAI from PyPI:
    ```sh
    pip install openvino-genai
    ```
2. Obtain model, e.g. export model to OpenVINO IR format from Hugging Face (see [Model Preparation Guide](https://openvinotoolkit.github.io/openvino.genai/docs/category/model-preparation) for more details):
    ```sh
    optimum-cli export openvino --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weight-format int4 --trust-remote-code TinyLlama_1_1b_v1_ov
    ```
3. Run inference:
    ```python
    import openvino_genai as ov_genai

    pipe = ov_genai.LLMPipeline("TinyLlama_1_1b_v1_ov", "CPU")  # Use CPU or GPU as devices without any other code change
    print(pipe.generate("What is OpenVINO?", max_new_tokens=100))
    ```

## Supported Generative AI Scenarios

OpenVINO™ GenAI library provides very lightweight C++ and Python APIs to run the following Generative AI Scenarios:
 - [Text generation using Large Language Models (LLMs)](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/text-generation/) - Chat with local Llama, Phi, Qwen and other models
 - [Image processing using Visual Language Model (VLMs)](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/) - Analyze images/videos with LLaVa, MiniCPM-V and other models
 - [Image generation using Diffusers](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-generation/) - Generate images with Stable Diffusion & Flux models
 - [Speech recognition using Whisper](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/speech-recognition/) - Convert speech to text using Whisper models
 - [Speech generation using SpeechT5](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/speech-generation/) - Convert text to speech using SpeechT5 TTS models
 - [Semantic search using Text Embedding](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/text-embedding) - Compute embeddings for documents and queries to enable efficient retrieval in RAG workflows
 - [Text Rerank for Retrieval-Augmented Generation (RAG)](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/text-rerank) - Analyze the relevance and accuracy of documents and queries for your RAG workflows

Library efficiently supports LoRA adapters for Text and Image generation scenarios:
- Load multiple adapters per model
- Select active adapters for every generation
- Mix multiple adapters with coefficients via alpha blending

All scenarios are run on top of OpenVINO Runtime that supports inference on CPU, GPU and NPU. See [here](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html) for platform support matrix.

<a id="optimization-methods"></a>

## Supported Generative AI Optimization Methods

OpenVINO™ GenAI library provides a transparent way to use state-of-the-art generation optimizations:
- Speculative decoding that employs two models of different sizes and uses the large model to periodically correct the results of the small model. See [here](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/) for more detailed overview
- KVCache token eviction algorithm that reduces the size of the KVCache by pruning less impacting tokens.
- Sparse attention, which accelerates prefill by attending only to the most important regions of the attention matrix. OpenVINO GenAI currently supports two modes: Tri-shape and XAttention. See [here](https://openvinotoolkit.github.io/openvino.genai/docs/concepts/optimization-techniques/sparse-attention-prefill) for more details.

Additionally, OpenVINO™ GenAI library implements a continuous batching approach to use OpenVINO within LLM serving. The continuous batching library could be used in LLM serving frameworks and supports the following features:
- Prefix caching that caches fragments of previous generation requests and corresponding KVCache entries internally and uses them in case of repeated query.

Continuous batching functionality is used within OpenVINO Model Server (OVMS) to serve LLMs, see [here](https://docs.openvino.ai/2025/openvino-workflow/model-server/ovms_what_is_openvino_model_server.html) for more details.


## Additional Resources

- [OpenVINO Generative AI workflow](https://docs.openvino.ai/2025/openvino-workflow-generative.html)
- [Optimum Intel and OpenVINO](https://huggingface.co/docs/optimum/intel/openvino/export)
- [OpenVINO Notebooks with GenAI](https://openvinotoolkit.github.io/openvino_notebooks/?libraries=OpenVINO+GenAI)

## License

The OpenVINO™ GenAI repository is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release
your contribution under these terms.
