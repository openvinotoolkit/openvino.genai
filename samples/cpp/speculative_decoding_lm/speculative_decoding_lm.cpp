// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (7 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <DRAFT_MODEL_DIR> '<PROMPT>' <GEN_LEN> <SMAPLING_TYPE> <SD_TYPE>");
    }

    std::string main_model_path = argv[1];
    std::string draft_model_path = argv[2];
    std::string prompt = argv[3];
    size_t gen_len = std::stoi(argv[4]);
    std::string sampling_type = argv[5];
    std::string sd_type = argv[6];

    if (sampling_type != "multinomial" && sampling_type != "greedy") {
        return -1;
    }

    if (sd_type != "static" && sd_type != "dynamic") {
        return -1;
    }

    ov::genai::GenerationConfig config = sampling_type == "multinomial" ? ov::genai::multinomial() : ov::genai::greedy();
    config.max_new_tokens = gen_len;

    if (draft_model_path != "none") {
        if (sd_type == "static") {
        // Speculative decoding generation parameters are mutually excluded
        // add parameter to enable speculative decoding to generate `num_assistant_tokens` candidates by draft_model per iteration
        config.num_assistant_tokens = 5;
        } else {
            // add parameter to enable speculative decoding to generate candidates by draft_model while candidate probability is higher than `assistant_confidence_threshold`
            config.assistant_confidence_threshold = 0.4;
        }
    }
    
    // User can run main and draft model on different devices.
    // Please, set device for main model in `LLMPipeline` constructor and in in `ov::genai::draft_model` for draft.
    std::string main_device = "CPU", draft_device = main_device;

    // Perform the inference
    auto get_default_block_size = [](const std::string& device) {
        const size_t cpu_block_size = 32;
        const size_t gpu_block_size = 16;

        bool is_gpu = device.find("GPU") != std::string::npos;

        return is_gpu ? gpu_block_size : cpu_block_size;
    };

    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.cache_size = 5;
    scheduler_config.block_size = get_default_block_size(main_device);

    auto streamer = [](std::string subword) {
        // std::cout << subword << std::flush;
        return false;
    };

    // Example to run main_model on GPU and draft_model on CPU:
    // ov::genai::LLMPipeline pipe(main_model_path, "GPU", ov::genai::draft_model(draft_model_path, "CPU"), ov::genai::scheduler_config(scheduler_config));
    if (draft_model_path != "none") {
        ov::genai::LLMPipeline pipe(main_model_path, main_device, ov::genai::draft_model(draft_model_path, draft_device), ov::genai::scheduler_config(scheduler_config));
        
        // Since the streamer is set, the results will
        // be printed each time a new token is generated.
        pipe.generate(prompt, config, streamer);
    } else {
        ov::genai::LLMPipeline pipe(main_model_path, main_device, ov::genai::scheduler_config(scheduler_config));
        
        // Since the streamer is set, the results will
        // be printed each time a new token is generated.
        pipe.generate(prompt, config, streamer);
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
