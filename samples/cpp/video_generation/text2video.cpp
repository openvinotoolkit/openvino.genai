// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>
#include <random>
#include <filesystem>

#include "progress_bar.hpp"

#include <openvino/genai/video_generation/text2video_pipeline.hpp>

int main(int32_t argc, char* argv[]) {
    OPENVINO_ASSERT(argc == 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>'");

    std::filesystem::path models_dir = argv[1];
    std::string prompt = argv[2];
    // TODO: Test GPU, NPU, HETERO, MULTI, AUTO, different steps on different devices
    const std::string device = "CPU";  // GPU can be used as well

    ov::genai::Text2VideoPipeline pipe(models_dir, device);
    pipe.m_impl->m_generation_config.num_frames = 1;
    ov::Tensor image = pipe.generate(
        prompt,
        "worst quality, inconsistent motion, blurry, jittery, distorted",
        ov::genai::height(512),
        ov::genai::width(704),
        ov::genai::num_inference_steps(50),
        ov::genai::num_images_per_prompt(1),
        ov::genai::callback(progress_bar)
    );
    return EXIT_SUCCESS;
// } catch (const std::exception& error) {
//     try {
//         std::cerr << error.what() << '\n';
//     } catch (const std::ios_base::failure&) {}
//     return EXIT_FAILURE;
// } catch (...) {
//     try {
//         std::cerr << "Non-exception object thrown\n";
//     } catch (const std::ios_base::failure&) {}
//     return EXIT_FAILURE;
}
