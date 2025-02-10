// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (2 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR>");
    }
    std::string prompt;
    std::string models_path = argv[1];

    std::string device = "CPU";  // GPU, NPU can be used as well
    ov::genai::LLMPipeline pipe(models_path, device);
    
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;

    auto streamer = [](std::string word) {
        std::cout << word << std::flush;
        // Return flag corresponds whether generation should be stopped.
        return ov::genai::StreamingStatus::RUNNING;
    };

    pipe.start_chat();
    std::cout << "question:\n";
    while (std::getline(std::cin, prompt)) {
        pipe.generate(prompt, config, streamer);
        std::cout << "\n----------\n"
            "question:\n";
    }
    pipe.finish_chat();
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
