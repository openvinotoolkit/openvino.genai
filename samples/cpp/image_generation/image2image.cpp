// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/image2image_pipeline.hpp"

#include "imwrite.hpp"
#include "load_image.hpp"
#include "progress_bar.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 3 && argc <= 4, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>' [ <IMAGE> ]");

    const std::string models_path = argv[1], prompt = argv[2];
    const std::string image_path = (argc > 3) ? argv[3] : "";
    const std::string device = "GPU";  // GPU can be used as well

    ov::genai::Image2ImagePipeline pipe(models_path, device);
    ov::Tensor generated_image;

    if (image_path.empty()) {
        generated_image = pipe.generate(prompt,
            ov::genai::width(512),
            ov::genai::height(512),
            ov::genai::num_inference_steps(20),
            ov::genai::num_images_per_prompt(1),
            ov::genai::strength(1.f),
            ov::genai::callback(progress_bar));
    } else {
        ov::Tensor image = utils::load_image(image_path);

        generated_image = pipe.generate(prompt, image,
            // controls how initial image is noised after being converted to latent space. `1` means initial image is fully noised
            ov::genai::strength(0.8f),
            ov::genai::callback(progress_bar));
    }

    // writes `num_images_per_prompt` images by pattern name
    imwrite("image_%d.bmp", generated_image, true);

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
