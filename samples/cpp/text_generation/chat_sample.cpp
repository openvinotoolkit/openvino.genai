// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    std::string models_path;
    std::string device = "CPU";  // GPU, NPU can be used as well
    std::string system_prompt;

    // Parse command-line arguments: <MODEL_DIR> [DEVICE] [--system_prompt "<text>"]
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--system_prompt" && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (models_path.empty()) {
            models_path = arg;
        } else if (device == "CPU") {
            device = arg;
        } else {
            throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                " <MODEL_DIR> [DEVICE] [--system_prompt \"<text>\"]");
        }
    }
    if (models_path.empty()) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
            " <MODEL_DIR> [DEVICE] [--system_prompt \"<text>\"]");
    }

    std::string prompt;
    ov::genai::LLMPipeline pipe(models_path, device);

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;

    auto streamer = [](std::string word) {
        std::cout << word << std::flush;
        // Return flag corresponds whether generation should be stopped.
        return ov::genai::StreamingStatus::RUNNING;
    };

    ov::genai::ChatHistory chat_history;
    if (!system_prompt.empty()) {
        chat_history.push_back({{"role", "system"}, {"content", system_prompt}});
    }

    std::cout << "question:\n";
    while (std::getline(std::cin, prompt)) {
        chat_history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
        ov::genai::DecodedResults decoded_results = pipe.generate(chat_history, config, streamer);
        chat_history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});
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
