# OpenVINO™ GenAI

The OpenVINO™ GenAI repository consists of GenAI library and additional GenAI samples.

## OpenVINO™ GenAI Library

OpenVINO™ GenAI is a new flavor of OpenVINO, aiming to simplify running inference of generative AI models.
It hides the complexity of the generation process and minimizes the amount of code required.

For installation and usage guide, refer to the [GenAI Library README](./src/README.md).

## OpenVINO™ GenAI Samples

The OpenVINO™ GenAI repository contains pipelines that implement image and text generation tasks.
The implementation uses OpenVINO capabilities to optimize the pipelines. Each sample covers
a family of models and suggests certain modifications to adapt the code to specific needs.
It includes the following pipelines:

1. [Benchmarking script for large language models](./llm_bench/python/)
2. [Text generation C++ samples that support most popular models like LLaMA 2](./text_generation/causal_lm/cpp/)
3. [Stable Diffuison (with LoRA) C++ image generation pipeline](./image_generation/stable_diffusion_1_5/cpp/)
4. [Latent Consistency Model (with LoRA) C++ image generation pipeline](./image_generation/lcm_dreamshaper_v7/cpp/)

### Requirements

Requirements may vary for different samples. See respective readme files for more details,
and make sure to install the OpenVINO version listed there. Refer to documentation to see
[how to install OpenVINO](https://docs.openvino.ai/install).

The supported devices are CPU and GPU including Intel discrete GPU.

See also: https://docs.openvino.ai/2023.3/gen_ai_guide.html.

## License

The OpenVINO™ GenAI repository is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release
your contribution under these terms.
