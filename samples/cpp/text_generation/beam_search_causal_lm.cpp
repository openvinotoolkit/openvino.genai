// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/genai/llm_pipeline.hpp>

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<PROMPT 1>' ['<PROMPT 2>' ...]");
    }
    auto prompts = std::vector<std::string>(argv + 2, argv + argc);
    std::string models_path = argv[1];

    std::string device = "CPU";  // GPU can be used as well
    ov::genai::LLMPipeline pipe(models_path, device);

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 20;
    config.num_beam_groups = 3;
    config.num_beams = 15;
    config.diversity_penalty = 1.0f;
    config.num_return_sequences = config.num_beams;

    auto beams = pipe.generate(prompts, config);
    std::cout << beams << '\n';
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
