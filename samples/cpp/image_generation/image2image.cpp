// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/image2image_pipeline.hpp"

#include "load_image.hpp"
#include "imwrite.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc == 4, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>' <IMAGE>");

    const std::string models_path = argv[1], prompt = argv[2], image_path = argv[3];
    const std::string device = "CPU";  // GPU can be used as well

    ov::Tensor image = utils::load_image(image_path);

    ov::genai::Image2ImagePipeline pipe(models_path, device);
    ov::Tensor generated_image = pipe.generate(prompt, image,
        // controls how initial image is noised after being converted to latent space. `1` means initial image is fully noised
        ov::genai::strength(0.8f));

    // writes `num_images_per_prompt` images by pattern name
    imwrite("image_%d.bmp", generated_image, true);
    auto perf_metrics = pipe.get_perfomance_metrics();
    std::cout << "pipeline generate duration ms:" << perf_metrics.generate_duration / 1000.0f << std::endl;
    std::cout << "pipeline inference duration ms:" << perf_metrics.get_inference_total_duration() << std::endl;
    std::cout << "pipeline iteration:" << perf_metrics.raw_metrics.iteration_durations.size() << std::endl;

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
