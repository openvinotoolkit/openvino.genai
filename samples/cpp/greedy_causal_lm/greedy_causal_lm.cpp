// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (3 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\" <GENERATED_LEN>");

    std::string model_path = argv[1];
    std::string prompt = argv[2];
    std::string len = argv[3];
    std::string device = "CPU";  // GPU can be used as well

    ov::genai::LLMPipeline pipe(model_path, device);
    ov::genai::GenerationConfig config;
    config.max_new_tokens = std::stoi(len);
    auto start_time = std::chrono::system_clock::now();
    std::string result = pipe.generate(prompt, config);
    auto end_time = std::chrono::system_clock::now();
    std::cout << result << std::endl;
    
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << std::endl;
    std::cout << "Duration: " << duration.count() << std::endl;
    std::cout << "Infer number: " << config.max_new_tokens << std::endl;
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
