// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>

ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}

int main(int argc, char* argv[]) try {
    if (argc < 3 || argc > 4) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES> <DEVICE>");
    }

    std::vector<ov::Tensor> rgbs = utils::load_images(argv[2]);

    // GPU and NPU can be used as well.
    // Note: If NPU is selected, only language model will be run on NPU
    std::string device = (argc == 4) ? argv[3] : "CPU";
    ov::AnyMap properties;
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        properties.insert({ov::cache_dir("vlm_cache")});
    }
    properties.insert({ov::genai::prompt_lookup(true)});

    ov::genai::VLMPipeline pipe(argv[1], device, properties);

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;
    // Define candidates number for candidate generation
    generation_config.num_assistant_tokens = 5;
    // Define max_ngram_size
    generation_config.max_ngram_size = 3;

    std::string prompt;

    ov::genai::ChatHistory history;
    
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
