// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/text2image_pipeline.hpp"

#include "imwrite.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc == 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>'");

    const std::string models_path = argv[1], prompt = argv[2];
    const std::string device = "CPU";  // GPU, NPU can be used as well

    auto txt2img_callback = [](size_t iter_num, ov::Tensor& latents) -> bool {
        std::cout << "txt2img_callback " << iter_num << latents.get_shape() << std::endl;
        std::cout << "main data " << std::endl;
        for (int i = 0; i < 10; ++i)
            std::cout << latents.data<float>()[i] << " ";
        std::cout << std::endl;
        if (iter_num == 3)
            return true;
        return false;
    };

    ov::genai::Text2ImagePipeline pipe(models_path, device);
    ov::Tensor image = pipe.generate(prompt,
        ov::genai::width(512),
        ov::genai::height(512),
        ov::genai::num_inference_steps(20),
        ov::genai::num_images_per_prompt(2)
        , ov::genai::callback(txt2img_callback)
        );
        // std::function<bool(size_t, ov::Tensor&)>

    // writes `num_images_per_prompt` images by pattern name
    imwrite("image_%d.bmp", image, true);

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
