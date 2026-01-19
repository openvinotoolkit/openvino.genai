// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (3 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\"");

    std::string models_path = argv[1];
    std::string prompt = argv[2];
    std::string device = "GPU";  // CPU can be used as well
    ov::AnyMap pipe_config = {};
    
    // Enable save_ov_model for GGUF files to save optimized model variant
    if (models_path.size() >= 5 && models_path.substr(models_path.size() - 5) == ".gguf") {
        ov::genai::SaveOVModelConfig save_config;
        save_config.mode = ov::genai::SaveOVModelConfig::SaveMode::ORIGINAL;
        pipe_config.insert(ov::genai::save_ov_model_config(save_config));
    } else {
        std::cout << "[TEST] Not a GGUF model, save_ov_model_config disabled" << std::endl;
    }
    
    std::cout << "[INFO] Creating LLMPipeline with device: " << device << std::endl;
    ov::genai::LLMPipeline pipe(models_path, device, pipe_config);
    std::cout << "[INFO] LLMPipeline created successfully" << std::endl;
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    std::string result = pipe.generate(prompt, config);
    std::cout << result << std::endl;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
