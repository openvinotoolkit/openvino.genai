// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (3 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\"");

    std::string model_path = argv[1];
    std::string prompt = argv[2];
    std::string device = "CPU";  // GPU can be used as well

    // Perform the inference
    auto get_default_block_size = [](const std::string& device) {
        const size_t cpu_block_size = 32;
        const size_t gpu_block_size = 16;

        bool is_gpu = device.find("GPU") != std::string::npos;

        return is_gpu ? gpu_block_size : cpu_block_size;
    };

    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.cache_size = 5;
    scheduler_config.block_size = get_default_block_size(device);

    // It's possible to construct a Tokenizer from a different path.
    // If the Tokenizer isn't specified, it's loaded from the same folder.
    // User can run main and draft model on different devices.
    // Please, set device for main model in `LLMPipeline` constructor and in in `ov::genai::draft_model` for draft.
    // Example to run main_model on GPU and draft_model on CPU:
    // ov::genai::LLMPipeline pipe(main_model_path, "GPU", ov::genai::draft_model(draft_model_path, "CPU"), ov::genai::scheduler_config(scheduler_config));
    ov::genai::LLMPipeline pipe(model_path, device, ov::genai::scheduler_config(scheduler_config));

    // ov::genai::LLMPipeline pipe(model_path, device);
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    std::string result = pipe.generate(prompt, config);
    std::cout << result << std::endl;
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
