# OpenVINO™ GenAI

The OpenVINO™ GenAI repository consists of the GenAI library and additional GenAI samples.

## OpenVINO™ GenAI Library

OpenVINO™ GenAI is a flavor of OpenVINO, aiming to simplify running inference of generative AI models.
It hides the complexity of the generation process and minimizes the amount of code required.

For installation and usage instructions, refer to the [GenAI Library README](./src/README.md).

## OpenVINO™ GenAI Samples

The OpenVINO™ GenAI repository contains pipelines that implement image and text generation tasks.
The implementation uses OpenVINO capabilities to optimize the pipelines. Each sample covers
a family of models and suggests certain modifications to adapt the code to specific needs.
It includes the following pipelines:

1. [Benchmarking script for large language models](./llm_bench/python/README.md)
2. Text generation samples that support most popular models like LLaMA 2:
   - Python:
     1. [beam_search_causal_lm](./samples/python/beam_search_causal_lm/README.md)
     1. [benchmark_genai](./samples/python/benchmark_genai/README.md)
     2. [chat_sample](./samples/python/chat_sample/README.md)
     3. [greedy_causal_lm](./samples/python/greedy_causal_lm/README.md)
     4. [multinomial_causal_lm](./samples/python/multinomial_causal_lm/README.md)
   - C++:
     1. [beam_search_causal_lm](./samples/cpp/beam_search_causal_lm/README.md)
     1. [benchmark_genai](./samples/cpp/benchmark_genai/README.md)
     2. [chat_sample](./samples/cpp/chat_sample/README.md)
     3. [continuous_batching_accuracy](./samples/cpp/continuous_batching_accuracy)
     4. [continuous_batching_benchmark](./samples/cpp/continuous_batching_benchmark)
     5. [greedy_causal_lm](./samples/cpp/greedy_causal_lm/README.md)
     6. [multinomial_causal_lm](./samples/cpp/multinomial_causal_lm/README.md)
     7. [prompt_lookup_decoding_lm](./samples/cpp/prompt_lookup_decoding_lm/README.md)
     8. [speculative_decoding_lm](./samples/cpp/speculative_decoding_lm/README.md)
3. [Stable Diffuison and Latent Consistency Model (with LoRA) C++ image generation pipeline](./samples/cpp/text2image/README.md)

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
