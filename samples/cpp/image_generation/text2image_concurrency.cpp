// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <iostream>

#include "openvino/genai/image_generation/text2image_pipeline.hpp"


#include "imwrite.hpp"
#include "progress_bar.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>' '<PROMPT>' ...");

    const std::string models_path = argv[1];
    const std::string device = "CPU";  // GPU and NPU can be used as well

    std::vector<std::thread> threads;
    std::vector<std::string> prompts;
    std::vector<ov::genai::Text2ImagePipeline> pipelines;

    for (int i = 2; i < argc; ++i)
        prompts.push_back(argv[i]);

    // Prepare initial pipeline and compiled models into device
    ov::AnyMap properties;
    if (device == "NPU") {
        // Define static shape and guidance scale for NPU
        const int num_images_per_prompt = 1;
        const int height = 512;
        const int width = 512;
        const float guidance_scale = 7.5f;

        pipelines.emplace_back(models_path);
        pipelines.back().reshape(num_images_per_prompt, height, width, guidance_scale);
        pipelines.back().compile(device);  // All models are compiled for NPU
        // pipelines.back().compile("NPU", "NPU", "GPU");  // Compile for NPU and GPU, if needed

        // Don't specify N, H, W, and guidance_scale in the properties map because they were made static
        properties = ov::AnyMap{ov::genai::num_inference_steps(2)};
    } else {
        pipelines.emplace_back(models_path, device);

        properties = ov::AnyMap{ov::genai::width(512),
                                ov::genai::height(512),
                                ov::genai::num_inference_steps(2),
                                ov::genai::num_images_per_prompt(1)};
    }

    // Clone pipeline for concurrent usage
    for (size_t i = 1; i < prompts.size(); ++i)
        pipelines.emplace_back(pipelines.begin()->clone());

    for (size_t i = 0; i < prompts.size(); ++i) {
        std::string prompt = prompts[i];
        auto& pipe = pipelines.at(i);

        std::cout << "Starting to generate with prompt: '" << prompt << "'..." << std::endl;

        threads.emplace_back([i, &pipe, prompt, &properties] () {
            ov::Tensor image = pipe.generate(prompt, properties);

            imwrite("image_" + std::to_string(i) + "_%d.bmp", image, true);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return EXIT_SUCCESS;
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
