GenAI contains pipelines that implement image and text generation tasks. The implementation exploits OpenVINO capabilities to optimize the pipelines. Each sample covers a family of models and suggests that its implementation can be modified to adapt for a specific need.

> [!NOTE]
> This project is not for production use.

The project includes the following pipelines:

1. [Benchmarking script for large language models](./llm_bench/python/)
2. [Casual LM](./text_generation/casual_lm/cpp/)
3. [OpenVINO Stable Diffuison (with LoRA) C++ pipeline](./image_generation/stable_diffusion_1_5/cpp/)
