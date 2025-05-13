// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (4 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <ADAPTER_SAFETENSORS_FILE> \"<PROMPT>\"");

    std::string models_path = argv[1];
    std::string adapter_path = argv[2];
    std::string prompt = argv[3];
    std::string device = "CPU";  // GPU can be used as well

    using namespace ov::genai;

    Adapter adapter(adapter_path);
    // Adapter adapter1("/home/alikh/projects/openvino.genai/llm_models/TinyPixel/tinyllama-lora/adapter_model.safetensors");
    // LLMPipeline pipe(models_path, device, adapters(adapter, adapter1));    // register all required adapters here
    LLMPipeline pipe(models_path, device, adapters(adapter));

    // Resetting config to set greedy behaviour ignoring generation config from model directory.
    // It helps to compare two generations with and without LoRA adapter.
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    pipe.set_generation_config(config);

    std::cout << "Generate with LoRA adapter and alpha set to 0.75:" << std::endl;
    // std::cout << pipe.generate(prompt, max_new_tokens(100), adapters({{adapter, 0.01}, {adapter1, 0.2}})) << std::endl;
    std::cout << pipe.generate(prompt, max_new_tokens(100), adapters({{adapter, 0.75}})) << std::endl;

    // std::cout << "\n-----------------------------";
    // std::cout << "Generate with LoRA adapter1 and alpha set to 0.75:" << std::endl;
    // std::cout << pipe.generate(prompt, max_new_tokens(100), adapters(adapter1, 0.75)) << std::endl;

    // std::cout << "\n-----------------------------";
    // std::cout << "\nGenerate without LoRA adapter:" << std::endl;
    // std::cout << pipe.generate(prompt, max_new_tokens(100), adapters()) << std::endl;

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
