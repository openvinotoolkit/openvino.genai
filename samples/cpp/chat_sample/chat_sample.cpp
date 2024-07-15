// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (2 != argc && 3 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <DEVICE>");
    }
    std::string prompt;
    std::string model_path = argv[1];
    std::string device = argc == 3 ? argv[2] : "CPU";
    ov::genai::LLMPipeline pipe(model_path, device);

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    std::function<bool(std::string)> streamer = [](std::string word) { 
        std::cout << word << std::flush;
        // Return flag corresponds whether generation should be stopped.
        // false means continue generation.
        return false; 
    };

    pipe.start_chat();
    for (;;) {
        std::cout << "question:\n";
        
        std::getline(std::cin, prompt);
        if (prompt == "Stop!") 
            break;

        pipe.generate(prompt, config, streamer);
        
        std::cout << "\n----------\n";
    }
    pipe.finish_chat();
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
