// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/vlm_pipeline.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>

bool callback(std::string&& subword) {
    return !(std::cout << subword);
}

int main(int argc, char* argv[]) try {
    if (3 != argc) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE>");
    }
    ov::Tensor image = utils::load_image(argv[2]);
    std::string device = "CPU";  // GPU can be used as well
    ov::AnyMap enable_compile_cache;
    if ("GPU" == device) {
        // Cache compile models on disks for GPU to save time on the
        // next run. It's not beneficial for CPU.
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
    }
    ov::genai::VLMPipeline pipe(argv[1], device, enable_compile_cache);
    pipe.start_chat();

    std::string prompt;
    std::cout << "question:\n";
    if (!std::getline(std::cin, prompt)) {
        throw std::runtime_error("std::cin failed");
    }
    pipe.generate({prompt, image}, callback);
    std::cout << "\n----------\n"
        "question:\n";
    size_t counter = 1;
    while (std::getline(std::cin, prompt)) {
        pipe.generate({prompt}, callback);
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
