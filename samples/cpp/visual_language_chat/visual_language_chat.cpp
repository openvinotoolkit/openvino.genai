// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>
#include <openvino/runtime/intel_gpu/properties.hpp>

bool print_subword(std::string&& subword) {
    return !(std::cout << subword << std::flush);
}

int main(int argc, char* argv[]) try {
    if (3 != argc) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES>");
    }

    std::vector<ov::Tensor> images = utils::load_images(argv[2]);

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 200;

    std::string device = "CPU";  // GPU can be used as well
    ov::AnyMap enable_compile_cache;
    if ("GPU" == device) {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
    }
    ov::genai::VLMPipeline pipe(argv[1], device, enable_compile_cache);
    std::string prompt;

    pipe.start_chat();
    std::cout << "question:\n";

    std::getline(std::cin, prompt);
    pipe.generate(prompt,
                  ov::genai::images(images),
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
