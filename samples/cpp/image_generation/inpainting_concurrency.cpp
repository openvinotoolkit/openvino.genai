// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <string>

#include "openvino/genai/image_generation/inpainting_pipeline.hpp"

#include "imwrite.hpp"
#include "load_image.hpp"
#include "progress_bar.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 5, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>' '<PROMPT>' ... <IMAGE> <MASK_IMAGE>");

    const std::string models_path = argv[1];
    const std::string device = "CPU";  // GPU can be used as well
    const std::string mask_image_path = argv[argc - 1];
    const std::string image_path = argv[argc - 2];
    ov::Tensor image = utils::load_image(image_path);
    ov::Tensor mask_image = utils::load_image(mask_image_path);

    std::vector<std::thread> threads;
    std::vector<std::string> prompts;
    std::vector<ov::genai::InpaintingPipeline> pipelines;

    for (int32_t i = 2; i < argc - 2; ++i)
        prompts.push_back(argv[i]);

    pipelines.emplace_back(models_path, device);

    // Clone pipeline for concurrent usage
    for (size_t i = 1; i < prompts.size(); ++i)
        pipelines.emplace_back(pipelines.begin()->clone());

    for (size_t i = 0; i < prompts.size(); ++i) {
        std::string prompt = prompts[i];
        auto& pipe = pipelines.at(i);

        std::cout << "Starting to generate with prompt: '" << prompt << "'..." << std::endl;

        threads.emplace_back([i, &pipe, prompt, image, mask_image] () {

            ov::Tensor generated_image = pipe.generate(prompt, image, mask_image,
                ov::genai::num_inference_steps(4));

            // writes `num_images_per_prompt` images by pattern name
            imwrite("image_" + std::to_string(i) + "_%d.bmp", generated_image, true);
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
