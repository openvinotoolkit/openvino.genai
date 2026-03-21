// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>

ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}

int main(int argc, char* argv[]) try {
    std::string model_dir;
    std::string image_dir;
    std::string device = "CPU";
    std::string lookup = "false";
    std::string system_prompt;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--system_prompt" || arg == "-sys") {
            if (i + 1 < argc) {
                system_prompt = argv[++i];
            } else {
                throw std::runtime_error("Error: --system_prompt requires a value.");
            }
        } else if (model_dir.empty()) {
            model_dir = arg;
        } else if (image_dir.empty()) {
            image_dir = arg;
        } else if (device == "CPU") {
            device = arg;
        } else {
            lookup = arg;
        }
    }

    if (model_dir.empty() || image_dir.empty()) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES> [DEVICE] [PROMPT_LOOKUP] [--system_prompt \"prompt text\"]");
    }

    std::vector<ov::Tensor> rgbs = utils::load_images(image_dir);

    // GPU and NPU can be used as well.
    // Note: If NPU is selected, only language model will be run on NPU
    bool prompt_lookup = (lookup == "true");
    // Prompt lookup decoding in VLM pipeline enforces ContinuousBatching backend
    ov::AnyMap properties = {ov::genai::prompt_lookup(prompt_lookup)};
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        properties.insert({ov::cache_dir("vlm_cache")});
    }

    ov::genai::VLMPipeline pipe(model_dir, device, properties);

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;
    if (prompt_lookup) {
        // Define candidates number for candidate generation
        generation_config.num_assistant_tokens = 5;
        // Define max_ngram_size
        generation_config.max_ngram_size = 3;
    }

    std::string prompt;

    ov::genai::ChatHistory history;
    if (!system_prompt.empty()) {
        history.push_back({{"role", "system"}, {"content", system_prompt}});
    }
    
    std::cout << "question:\n";
    std::getline(std::cin, prompt);

    history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
    ov::genai::VLMDecodedResults decoded_results = pipe.generate(
        history,
        ov::genai::images(rgbs),
        ov::genai::generation_config(generation_config),
        ov::genai::streamer(print_subword)
    );
    history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});
    std::cout << "\n----------\n"
                 "question:\n";
    while (std::getline(std::cin, prompt)) {
        history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
        // New images and videos can be passed at each turn
        ov::genai::VLMDecodedResults decoded_results = pipe.generate(
            history,
            ov::genai::generation_config(generation_config),
            ov::genai::streamer(print_subword)
        );
        history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});
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
