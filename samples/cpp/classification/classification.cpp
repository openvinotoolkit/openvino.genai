// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../visual_language_chat/load_image.hpp"
#include "openvino/genai/classification/classification_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<TEXT>' [<IMAGE_FILE> ...]");
    }

    const std::string models_path = argv[1];
    const std::string text = argv[2];
    std::vector<ov::Tensor> images;
    for (int arg_idx = 3; arg_idx < argc; ++arg_idx) {
        images.push_back(utils::load_image(argv[arg_idx]));
    }

    const std::string device = "CPU";  // GPU can be used as well

    ov::genai::ClassificationPipeline pipeline(models_path, device);

    // Images are referenced in the text by <ov_genai_image_N> tags; untagged images are placed as in VLMPipeline.
    const ov::genai::ClassificationResult result = pipeline.classify(text, images);

    std::cout << std::fixed << std::setprecision(4);
    const std::vector<float>& scores = result.scores.at(0);
    for (size_t label_id = 0; label_id < scores.size(); ++label_id) {
        std::cout << label_id << ": " << scores[label_id] << '\n';
    }
    std::cout << std::defaultfloat;

} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
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
