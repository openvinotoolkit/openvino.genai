// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"
#include <regex>
#include <fstream>

int main(int argc, char* argv[]) try {
    if (2 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR>");
    }
    std::string prompt;
    std::string models_path = argv[1];

    std::string model_path = models_path + "/openvino_model.xml";
    std::string weights_path = std::regex_replace(model_path, std::regex(".xml"), ".bin");
    std::ifstream model_file(model_path, std::ios::binary | std::ios::ate);
    std::ifstream weights_file(weights_path, std::ios::binary | std::ios::ate);

    if (!model_file.is_open() || !weights_file.is_open()) {
        throw std::runtime_error("Cannot open model or weights file");
    }

    std::streamsize model_size = model_file.tellg();
    std::streamsize weights_size = weights_file.tellg();

    model_file.seekg(0, std::ios::beg);
    weights_file.seekg(0, std::ios::beg);

    std::vector<char> model_buffer(model_size);
    std::vector<char> weights_buffer(weights_size);

    if (!model_file.read(model_buffer.data(), model_size) || !weights_file.read(weights_buffer.data(), weights_size)) {
        throw std::runtime_error("Error reading model or weights file");
    }
    std::vector<uint8_t> model_uint8_buffer(model_buffer.begin(), model_buffer.end());
    std::vector<uint8_t> weights_uint8_buffer(weights_buffer.begin(), weights_buffer.end());
    

    std::string device = "CPU";  // GPU, NPU can be used as well
    // ov::genai::LLMPipeline pipe(models_path, device);
    
    ov::genai::Tokenizer tok(models_path);
    ov::genai::LLMPipeline pipe(model_uint8_buffer, weights_uint8_buffer, tok, device);

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    std::function<bool(std::string)> streamer = [](std::string word) { 
        std::cout << word << std::flush;
        // Return flag corresponds whether generation should be stopped.
        // false means continue generation.
        return false; 
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
