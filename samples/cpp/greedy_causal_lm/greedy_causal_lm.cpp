// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (3 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <ADAPTER_SAFETENSORS> \"<PROMPT>\"");

    std::string model_path = argv[1];
    std::string adapter_path = argv[2];
    std::string prompt = argv[3];
    std::string device = "CPU";  // GPU can be used as well

#if 1
    ov::genai::Adapter adapter(adapter_path, /*alpha = */ 1);
    ov::genai::AdaptersConfig adapters_config(adapter);
    ov::genai::LLMPipeline pipe(model_path, device, adapters_config);

    ov::genai::GenerationConfig config;
    config.adapters = adapters_config;  // FIXME: Should be redundant
    config.max_new_tokens = 100;
    std::cout << "Generation with LoRA adapter applied:\n";
    std::string result = pipe.generate(prompt, config);
    std::cout << result << std::endl;

    config.adapters.alphas.push_back(0);
    std::cout << "-------------------------------\n";
    std::cout << "Generation without LoRA adapter:\n";
    result = pipe.generate(prompt, config);
    std::cout << result << std::endl;
#else

    ov::genai::LLMPipeline pipe(model_path, device);

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    std::string result = pipe.generate(prompt, config);
    std::cout << result << std::endl;

#endif

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
