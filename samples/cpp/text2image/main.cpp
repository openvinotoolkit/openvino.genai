// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/image2image_pipeline.hpp"

#include "load_image.hpp"
#include "imwrite.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc == 4, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>' '<IMAGE>'");

    const std::string models_path = argv[1], prompt = argv[2];
    const std::string device = "CPU";  // GPU, NPU can be used as well

    std::vector<ov::Tensor> rgbs = utils::load_images(argv[3]);
    ov::Tensor rgb = rgbs[0];

    ov::Shape shape = rgb.get_shape();
    size_t height = shape[1], width = shape[2];
    // width = (width / 8) * 8;
    // height = (height / 8) * 8;

    // rgb = ov::Tensor(rgb.get_element_type(), { shape[0], height, width, shape[3] }, rgb.data());

    ov::genai::Image2ImagePipeline pipe(models_path, device);
    ov::Tensor image = pipe.generate(prompt, rgb,
        ov::genai::width(width),
        ov::genai::height(height),
        ov::genai::strength(0.4f),
        ov::genai::num_inference_steps(20),
        ov::genai::num_images_per_prompt(1));

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
