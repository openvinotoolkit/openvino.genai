// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>

bool print_subword(std::string&& subword) {
    return !(std::cout << subword << std::flush);
}

int main(int argc, char* argv[]) try {
    if (3 > argc || argc > 7) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE> [<DEVICE>] [<ENABLE_CDPRUNER>] [<NUM_VISUAL_TOKENS>] [<PRUNING_DEBUG_MODE>]");
    }

    std::string model_dir = argv[1];
    std::string image_file = argv[2];
    std::string device = argc > 3 ? argv[3] : "CPU";
    bool enable_cdpruner = argc > 4 ? (std::string(argv[4]) == "true" || std::string(argv[4]) == "1") : false;
    size_t visual_tokens_retain_percentage = argc > 5 ? std::stoul(argv[5]) : 30;
    bool pruning_debug_mode = argc > 6 ? (std::string(argv[6]) == "true" || std::string(argv[6]) == "1") : false;

    std::vector<ov::Tensor> rgbs = utils::load_images(image_file);

    // GPU and NPU can be used as well.
    // Note: If NPU is selected, only language model will be run on NPU
    ov::AnyMap enable_compile_cache;
    if (device == "GPU") {
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
    }
    
    // Initialize VLMPipeline with cache configuration if needed
    ov::genai::VLMPipeline pipe(model_dir, device, enable_compile_cache);
    
    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;
    // Configure CDPruner if requested
    if (enable_cdpruner) {
        std::cout << "Enabling CDPruner with keeping " << visual_tokens_retain_percentage << "% visual tokens" << std::endl;
        generation_config.enable_pruning = enable_cdpruner;
        generation_config.visual_tokens_retain_percentage = visual_tokens_retain_percentage;
        generation_config.pruning_debug_mode = pruning_debug_mode;
    }


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
