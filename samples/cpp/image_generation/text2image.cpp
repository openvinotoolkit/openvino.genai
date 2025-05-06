// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <iostream>

#include "openvino/genai/image_generation/text2image_pipeline.hpp"


#include "imwrite.hpp"
#include "progress_bar.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc == 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>'");  // TODO: <PROMPT> unused

    const std::string models_path = argv[1], prompt = argv[2];
    const std::string device = "CPU";  // GPU can be used as well

    std::vector<std::string> prompts = {
        "happy dog",
        "black cat",
        "yellow raspberry",
        "retro personal computer",
        "walking astronaut",
        "fish with a hat",
        "flying car",
    };

    ov::genai::Text2ImagePipeline pipe(models_path, device);
    
    std::vector<std::thread> threads;

    for (size_t i = 0; i < prompts.size(); ++i) {
        const std::string p = prompts[i];
        threads.emplace_back([i, &pipe, p] () {
            std::cout << "Generating... " << i << std::endl;
            ov::genai::Text2ImagePipeline::GenerationRequest request = pipe.create_generation_request();
            ov::Tensor image = request.generate(p,
                ov::AnyMap{
                    ov::genai::width(512),
                    ov::genai::height(512),
                    ov::genai::num_inference_steps(20),
                    ov::genai::num_images_per_prompt(1)});
            std::cout << "Generated " << i << std::endl;
            imwrite("mt_image_512" + std::to_string(i) + "_%d.bmp", image, true);
            std::cout << "Generation saved" << std::endl;
        });
    }

    // join all threads
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
