// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <string>

#include "openvino/genai/image_generation/image2image_pipeline.hpp"

#include "imwrite.hpp"
#include "load_image.hpp"
#include "progress_bar.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 4, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>' '<PROMPT>' ... <IMAGE>");

    const std::string models_path = argv[1];//, prompt = argv[2], image_path = argv[3];

    std::vector<std::string> prompts;
    for (int32_t i = 2; i < argc - 1; ++i) {
        prompts.push_back(argv[i]);
    }

    const std::string image_path = argv[argc - 1];

    const std::string device = "CPU";  // GPU can be used as well

    ov::Tensor image = utils::load_image(image_path);

    ov::genai::Image2ImagePipeline pipe(models_path, device);

    std::vector<std::thread> threads;

    for (size_t i = 0; i < prompts.size(); ++i) {
        threads.emplace_back([i, &pipe, models_path, prompts, image] () {
            std::cout << "Starting to generate with prompt: '" << prompts[i] << "'..." << std::endl;

            ov::genai::Image2ImagePipeline request = pipe.clone();
            
            ov::Tensor generated_image = request.generate(prompts[i], image,
                // controls how initial image is noised after being converted to latent space. `1` means initial image is fully noised
                ov::genai::strength(0.8f),
                ov::genai::num_inference_steps(4));

            // writes `num_images_per_prompt` images by pattern name
            std::cout << "Generated image for prompt: '" << prompts[i] << "', saving..." << std::endl;
            imwrite("image_" + std::to_string(i) + "_%d.bmp", generated_image, true);
    
            std::cout << "Saved image for prompt: '" << prompts[i] << "'" << std::endl;
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
