// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (argc < 2 || argc > 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <DEVICE>");
    }
    std::string prompt;
    std::string models_path = argv[1];

    // Default device is CPU; can be overridden by the second argument
    std::string device = (argc == 3) ? argv[2] : "CPU";  // GPU, NPU can be used as well
    ov::genai::LLMPipeline pipe(models_path, device);
    
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;

    auto streamer = [](std::string word) {
        std::cout << word << std::flush;
        // Return flag corresponds whether generation should be stopped.
        return ov::genai::StreamingStatus::RUNNING;
    };

    ov::genai::ChatHistory chat_history;

    std::cout << "question:\n";
    while (std::getline(std::cin, prompt)) {
        chat_history.push_back({{"role", "user"}, {"content", prompt}});
        ov::genai::DecodedResults decoded_results = pipe.generate(chat_history, config, streamer);
        std::string output = decoded_results.texts[0];
        chat_history.push_back({{"role", "assistant"}, {"content", output}});
        std::cout << "\n----------\n"
            "question:\n";
    }
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
