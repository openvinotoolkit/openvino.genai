// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (argc < 3 || argc > 4) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<PROMPT>' [RNG_SEED]");
    }

    std::string models_path = argv[1];
    std::string prompt = argv[2];

    std::string device = "CPU";  // GPU can be used as well
    ov::genai::LLMPipeline pipe(models_path, device);

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    config.do_sample = true;
    config.top_p = 0.9;
    config.top_k = 30;

    if (argc == 4) {
        char* end = nullptr;
        const char* seed_str = argv[3];
        if (seed_str[0] == '-') {
            throw std::runtime_error("RNG_SEED must be a non-negative integer.");
        }
        const unsigned long long seed = std::strtoull(seed_str, &end, 10);
        if (*end != '\0' || seed > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("RNG_SEED must be a non-negative integer.");
        }
        config.rng_seed = static_cast<size_t>(seed);
    }

    auto streamer = [](std::string subword) {
        std::cout << subword << std::flush;
        return ov::genai::StreamingStatus::RUNNING;
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
