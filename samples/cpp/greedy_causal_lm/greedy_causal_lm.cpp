// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (3 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> [<ADAPTER_SAFETENSORS>] \"<PROMPT>\"");

    std::string model_path = argv[1];
    std::string adapter_path = argc > 3 ? argv[2] : "";
    std::string prompt = argv[argc - 1];
    std::string device = "CPU";  // GPU can be used as well

    ov::genai::Adapter adapter;
    ov::genai::AdaptersConfig adapters_config;
    if(!adapter_path.empty()) {
        adapter = ov::genai::Adapter(adapter_path);
        adapters_config.set(adapter, 1.0);
    }
    ov::genai::LLMPipeline pipe(model_path, device, adapters_config);

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;

    if(!adapter_path.empty()) { 
        config.adapters = adapters_config;  // FIXME: Should be redundant
        std::cout << "Generation with LoRA adapter applied:\n";
        std::string result = pipe.generate(prompt, config);
        std::cout << result << std::endl;

        // Disable adapter to compare with generation without the adapter
        config.adapters.set(adapter, 0.0);
        
        std::cout << "-------------------------------\n";
        std::cout << "Generation without LoRA adapter:\n";
    }

    std::string result = pipe.generate(prompt, config);
    std::cout << result << std::endl;

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
