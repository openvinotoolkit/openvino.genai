// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (3 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <DRAFT_MODEL_DIR> '<PROMPT>'");
    }

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    // Speculative decoding generation parameters like `num_assistant_tokens` and `assistant_confidence_threshold` are mutually excluded
    // add parameter to enable speculative decoding to generate `num_assistant_tokens` candidates by draft_model per iteration
    config.num_assistant_tokens = 5;

    std::string model_path = argv[1];
    // std::string prompt = argv[2];
    std::string prompt = R"(from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
""" Check if in given list of numbers, are any two numbers closer to each other than
given threshold.
>>> has_close_elements([1.0, 2.0, 3.0], 0.5)
False
>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
True
""")";
    
    // User can run main and draft model on different devices.
    // Please, set device for main model in `LLMPipeline` constructor and in in `ov::genai::draft_model` for draft.
    std::string device = "CPU";

    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.cache_size = 5;

    // Different devices require different block sizes, so different scheduler configs need to be set.
    ov::genai::LLMPipeline pipe(
        model_path,
        device,
        ov::genai::enable_prompt_lookup(true),
        ov::genai::scheduler_config(scheduler_config));

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
