// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>

bool print_subword(std::string&& subword) {
    return !(std::cout << subword << std::flush);
}

int main(int argc, char* argv[]) try {
    if (3 > argc || argc > 6) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE> [<DEVICE>] [<PRUNING_RATIO>] [PRUNING_RELEVANCE_WEIGHT]");
    }

    std::string model_dir = argv[1];
    std::string image_file = argv[2];
    std::string device = argc > 3 ? argv[3] : "CPU";
    size_t pruning_ratio = argc > 4 ? std::stoul(argv[4]) : 0;  // 0 means disabled
    float pruning_relevance_weight = argc > 5 ? std::stof(argv[5]) : 0.5f;

    std::vector<ov::Tensor> rgbs = utils::load_images(image_file);

    // GPU and NPU can be used as well.
    // Note: If NPU is selected, only language model will be run on NPU
    ov::AnyMap enable_compile_cache;
    if (device == "GPU") {
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
    }

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;
    // Configure CDPruner if requested
    if (pruning_ratio == 0) {
        std::cout << "[CDPruner] Disabled" << std::endl;
    } else if (pruning_ratio > 0 && pruning_ratio < 100) {
        std::cout << "[CDPruner] Enabling CDPruner with " << pruning_ratio << "% visual token pruning" << std::endl;
        generation_config.pruning_ratio = pruning_ratio;
        generation_config.relevance_weight = pruning_relevance_weight;
    } else {
        std::cout << "[CDPruner] Invalid pruning ratio(" << pruning_ratio << "%). Disabling CDPruner." << std::endl;
    }

    // Initialize VLMPipeline with cache configuration if needed
    ov::genai::VLMPipeline pipe(model_dir, device, enable_compile_cache);

    std::string prompt;

    pipe.start_chat();
    std::cout << "question:\n";

    std::getline(std::cin, prompt);
    pipe.generate(prompt,
                  ov::genai::images(rgbs),
                  ov::genai::generation_config(generation_config),
                  ov::genai::streamer(print_subword));
    std::cout << "\n----------\n"
        "question:\n";
    while (std::getline(std::cin, prompt)) {
        pipe.generate(prompt,
                      ov::genai::generation_config(generation_config),
                      ov::genai::streamer(print_subword));
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
