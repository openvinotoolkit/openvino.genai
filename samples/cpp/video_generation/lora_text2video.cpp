// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>
#include <random>
#include <filesystem>

#include "progress_bar.hpp"
#include "imwrite_video.hpp"

#include <openvino/genai/video_generation/text2video_pipeline.hpp>

int main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 3 && (argc - 3) % 2 == 0, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>' [<LORA_SAFETENSORS> <ALPHA> ...]]");

    std::filesystem::path models_dir = argv[1];
    std::string prompt = argv[2];

    const std::string device = "CPU";  // GPU can be used as well
    float frame_rate = 25.0f;

    ov::genai::AdapterConfig adapter_config;
    // Multiple LoRA adapters applied simultaneously are supported, parse them all and corresponding alphas from cmd parameters:
    for(size_t i = 0; i < (argc - 3)/2; ++i) {
        ov::genai::Adapter adapter(argv[3 + 2*i]);
        float alpha = std::atof(argv[3 + 2*i + 1]);
        adapter_config.add(adapter, alpha);
    }

    // LoRA adapters passed to the constructor will be activated by default in next generates
    ov::genai::Text2VideoPipeline pipe(models_dir, device, ov::genai::adapters(adapter_config));

    std::cout << "Generating video with LoRA adapters applied, resulting video will be in lora_video.avi\n";
    auto output = pipe.generate(
        prompt,
        ov::genai::negative_prompt("worst quality, inconsistent motion, blurry, jittery, distorted"),
        ov::genai::height(480),
        ov::genai::width(704),
        ov::genai::num_frames(161),
        ov::genai::num_inference_steps(25),
        ov::genai::num_videos_per_prompt(1),
        ov::genai::callback(progress_bar),
        ov::genai::frame_rate(frame_rate),
        ov::genai::guidance_scale(3)
    );

    save_video("lora_video.avi", output.video, frame_rate);

    std::cout << "Generating video without LoRA adapters applied, resulting video will be in baseline_video.avi\n";
    output = pipe.generate(
        prompt,
        ov::genai::adapters(),  // passing adapters in generate overrides adapters set in the constructor; adapters() means no adapters
        ov::genai::negative_prompt("worst quality, inconsistent motion, blurry, jittery, distorted"),
        ov::genai::height(480),
        ov::genai::width(704),
        ov::genai::num_frames(161),
        ov::genai::num_inference_steps(25),
        ov::genai::num_videos_per_prompt(1),
        ov::genai::callback(progress_bar),
        ov::genai::frame_rate(frame_rate),
        ov::genai::guidance_scale(3)
    );

    save_video("baseline_video.avi", output.video, frame_rate);

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
