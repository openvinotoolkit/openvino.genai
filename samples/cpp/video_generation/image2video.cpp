// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <string>

#include "load_image.hpp"
#include "progress_bar.hpp"
#include "imwrite_video.hpp"

#include <openvino/genai/video_generation/image2video_pipeline.hpp>

int main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc == 4, "Usage: ", argv[0], " <MODEL_DIR> <IMAGE_PATH> '<PROMPT>'");

    std::filesystem::path models_dir = argv[1];
    std::filesystem::path image_path = argv[2];
    std::string prompt = argv[3];

    const std::string device = "CPU";  // GPU can be used as well
    float frame_rate = 25.0f;

    ov::Tensor image = utils::load_image(image_path);

    ov::genai::Image2VideoPipeline pipe(models_dir, device);
    auto output = pipe.generate(
        image,
        prompt,
        ov::genai::negative_prompt("static, motionless, frozen, still photograph, no movement, low quality, blurry, distorted"),
        ov::genai::height(480),
        ov::genai::width(704),
        ov::genai::num_frames(161),
        ov::genai::num_inference_steps(50),
        ov::genai::num_videos_per_prompt(1),
        ov::genai::callback(progress_bar),
        ov::genai::frame_rate(frame_rate),
        ov::genai::guidance_scale(4.0f)
    );

    save_video("genai_video.avi", output.video, frame_rate);

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
