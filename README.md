# OpenVINO™ GenAI

OpenVINO™ GenAI is a library of most popular Generative model pipelines, optimized execution methods and samples that runs on top of highly performant [OpenVINO Runtime](https://github.com/openvinotoolkit/openvino).

## Supported Generative AI scenarios

OpenVINO™ GenAI library provides very lightweight C++ and Python APIs to run following Generative Scenarios:
 - Text generation using Large Language Models. For example, chat with local LLaMa model
 - Image generation using Diffuser models, for example generation using Stable Diffusion models
 - Speech recognition using Whisper family models
 - Text generation using Large Visual Models, for instance Image analysis using LLaVa models family

All scenarios could be run on top of OpenVINO Runtime that supports inference using CPU, GPU and NPU. See [here](https://github.com/openvinotoolkit/openvino) for platform support matrix.

## Supported Generative AI optimization methods

OpenVINO™ GenAI library provides transparent way to use state of the art generation optimizations:
- Prefix caching that caches fragments of previous generation requests and corresponding KVCache entries internally and uses them in case of repeated query. See [here](https://google.com) for more detailed overview
- Speculative decoding that employs two models of different size and uses large model to periodically correct results of small model. See [here](https://google.com) for more detailed overview
- Lookahead decoding that attempts to guess multiple tokens based on historical context and corrects this guess using LLMs. See [here](https://google.com) for more detailed overview
- KVCache eviction algorithm that reduces size of the KVCache by pruning less impacting tokens. See [here](https://google.com) for more detailed overview

OpenVINO™ GenAI library implements continuous batching approach that allows processing multiple generation requests simultaneously and efficiently use compute resources. 

## Getting started

For installation and usage instructions, refer to the [GenAI Library README](./src/README.md).

## License

The OpenVINO™ GenAI repository is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release
your contribution under these terms.
