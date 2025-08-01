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
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE> [<DEVICE>] [<ENABLE_CDPRUNER>] [<NUM_VISUAL_TOKENS>]");
    }

    std::string model_dir = argv[1];
    std::string image_file = argv[2];
    std::string device = argc > 3 ? argv[3] : "CPU";
    bool enable_cdpruner = argc > 4 ? (std::string(argv[4]) == "true" || std::string(argv[4]) == "1") : false;
    size_t num_visual_tokens = argc > 5 ? std::stoul(argv[5]) : 64;

    std::vector<ov::Tensor> rgbs = utils::load_images(image_file);

    // GPU and NPU can be used as well.
    // Note: If NPU is selected, only language model will be run on NPU
    ov::AnyMap enable_compile_cache;
    if (device == "GPU") {
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
    }
    
    // Initialize VLMPipeline with cache configuration if needed
    ov::genai::VLMPipeline pipe(model_dir, device, enable_compile_cache);
    
    // Configure CDPruner if requested
    if (enable_cdpruner) {
        std::cout << "Enabling CDPruner with " << num_visual_tokens << " visual tokens" << std::endl;
        pipe.set_visual_token_pruning_config(
            num_visual_tokens,  // num_visual_tokens
            0.5f,              // relevance_weight  
            true               // enable_pruning
        );
        
        // Print current configuration
        auto config = pipe.get_visual_token_pruning_config();
        std::cout << "CDPruner configuration:" << std::endl;
        std::cout << "  - Enabled: " << (pipe.is_visual_token_pruning_enabled() ? "true" : "false") << std::endl;
        std::cout << "  - Num visual tokens: " << config["num_visual_tokens"].as<size_t>() << std::endl;
        std::cout << "  - Relevance weight: " << config["relevance_weight"].as<float>() << std::endl;
    } else {
        std::cout << "CDPruner is disabled" << std::endl;
        pipe.set_visual_token_pruning_enabled(false);
    }

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 3000;

    std::string prompt = "describe this image in details";

    pipe.start_chat();
    //std::cout << "question:\n";

    //std::getline(std::cin, prompt);
    pipe.generate(prompt,
                  ov::genai::images(rgbs),
                  ov::genai::generation_config(generation_config),
                  ov::genai::streamer(print_subword));
    //std::cout << "\n----------\n"
    //    "question:\n";
    //while (std::getline(std::cin, prompt)) {
    //    pipe.generate(prompt,
    //                  ov::genai::generation_config(generation_config),
    //                  ov::genai::streamer(print_subword));
    //    std::cout << "\n----------\n"
    //        "question:\n";
    //}
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
