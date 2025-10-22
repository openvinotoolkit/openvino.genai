// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <iostream>
#include <openvino/genai/visual_language/pipeline.hpp>

#include "load_image.hpp"

bool print_subword(std::string&& subword) {
    return !(std::cout << subword << std::flush);
}

int main(int argc, char* argv[]) try {
    if (argc < 3 || argc > 5) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE> [PROMPT] [DEVICE]");
    }

    std::string model_dir = argv[1];
    std::string image_path = argv[2];
    std::string prompt = (argc >= 4) ? argv[3] : "Describe the image in detail.";
    std::string device = (argc == 5) ? argv[4] : "CPU";

    std::cout << "Loading image from: " << image_path << std::endl;
    std::vector<ov::Tensor> rgbs = utils::load_images(image_path);
    std::cout << "Number of images loaded: " << rgbs.size() << std::endl;

    // Configure device-specific options
    ov::AnyMap device_config;
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the next run
        device_config.insert({ov::cache_dir("qwen2vl_cache")});
    }

    std::cout << "Initializing Qwen2.5-VL pipeline on " << device << "..." << std::endl;
    ov::genai::VLMPipeline pipe(model_dir, device, device_config);

    // Configure generation parameters
    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 256;  // Increased for more detailed descriptions

    std::cout << "\n========================================" << std::endl;
    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Response: ";

    // Generate response for the image
    pipe.start_chat();
    pipe.generate(prompt,
                  ov::genai::images(rgbs),
                  ov::genai::generation_config(generation_config),
                  ov::genai::streamer(print_subword));
    pipe.finish_chat();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Image processing completed successfully." << std::endl;

    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << "Error: " << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}
