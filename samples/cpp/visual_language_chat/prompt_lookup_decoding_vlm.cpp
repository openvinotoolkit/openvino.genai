// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "load_image.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (4 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES> <PROMPT>");
    }

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    // Define candidates number for candidate generation
    config.num_assistant_tokens = 5;
    // Define max_ngram_size
    config.max_ngram_size = 3;

    std::string model_path = argv[1];
    std::vector<ov::Tensor> rgbs = utils::load_images(argv[2]);
    std::string prompt = argv[3];

    std::string device = "CPU";

    ov::genai::VLMPipeline pipe(
        model_path,
        device,
        ov::genai::prompt_lookup(true));

    auto streamer = [](std::string subword) {
        std::cout << subword << std::flush;
        return ov::genai::StreamingStatus::RUNNING;
    };

    // Since the streamer is set, the results are
    // printed each time a new token is generated.
    pipe.generate(prompt, ov::genai::images(rgbs), ov::genai::generation_config(config), ov::genai::streamer(streamer));
    std::cout << std::endl;
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
