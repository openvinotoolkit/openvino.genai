// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (4 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<PROMPT>' <GEN_LEN>");
    }

    std::string model_path = argv[1];
    std::string prompt = argv[2];
    size_t gen_len = std::stoi(argv[3]);

    std::string device = "CPU";  // GPU can be used as well
    ov::genai::LLMPipeline pipe(model_path, device);

    ov::genai::GenerationConfig config;
    config.max_new_tokens = gen_len;
    // config.max_new_tokens = 100;
    config.do_sample = true;
    config.top_p = 0.9;
    config.top_k = 30;
    auto streamer = [](std::string subword) {
        std::cout << subword << std::flush;
        return false;
    };

    // Since the streamer is set, the results will
    // be printed each time a new token is generated.
    pipe.generate(prompt, config, streamer);
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
