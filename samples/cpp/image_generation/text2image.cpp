// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <iostream>

#include "openvino/genai/image_generation/text2image_pipeline.hpp"


#include "imwrite.hpp"
#include "progress_bar.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc == 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>'");

    const std::string models_path = argv[1], prompt = argv[2];
    const std::string device = "CPU";  // GPU can be used as well

    ov::genai::Text2ImagePipeline pipe(models_path, device);
    std::thread t([&pipe, prompt] () {

        std::cout << "Generating..." << std::endl;
        ov::Tensor image = pipe.generate("happy dog",
            ov::AnyMap{
                ov::genai::width(512),
                ov::genai::height(512),
                ov::genai::num_inference_steps(20),
                ov::genai::num_images_per_prompt(1)},
            
            1);
            //ov::genai::callback(progress_bar));

        // writes `num_images_per_prompt` images by pattern name
        imwrite("image_2_%d.bmp", image, true);
        std::cout << "Generation ended" << std::endl;

    });

    std::thread t1([&pipe, prompt] () {

        std::cout << "Generating..." << std::endl;
        ov::Tensor image = pipe.generate("black cat",
            ov::AnyMap{
                ov::genai::width(512),
                ov::genai::height(512),
                ov::genai::num_inference_steps(20),
                ov::genai::num_images_per_prompt(1)},
            
            2);
            //ov::genai::callback(progress_bar));

        // writes `num_images_per_prompt` images by pattern name
        imwrite("image_1_%d.bmp", image, true);
        std::cout << "Generation ended" << std::endl;

    });

    t.join();
    t1.join();


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
