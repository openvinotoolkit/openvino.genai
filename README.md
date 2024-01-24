## GenAI Pipeline Repository

The GenAI repository contains pipelines that implement image and text generation tasks.
The implementation uses OpenVINO capabilities to optimize the pipelines. Each sample covers
a family of models and suggests certain modifications to adapt the code to specific needs.
It includes the following pipelines:

1. [Benchmarking script for large language models](./llm_bench/python/)
2. [Text generation C++ samples that support most popular models like LLaMA 2](./text_generation/causal_lm/cpp/)
3. [Stable Diffuison (with LoRA) C++ image generation pipeline](./image_generation/stable_diffusion_1_5/cpp/)
4. [Latent Consistency Model (with LoRA) C++ image generation pipeline](./image_generation/lcm_dreamshaper_v7/cpp/)

### License

The GenAI repository is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release
your contribution under these terms.

## Requirements

Requirements may vary for different samples. See respective readme files for more details,
and make sure to install the OpenVINO version listed there. Refer to documentation to see
[how to install OpenVINO](https://docs.openvino.ai/install).

The supported devices are CPU and GPU including Intel discrete GPU.
