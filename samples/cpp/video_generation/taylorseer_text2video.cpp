// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <iostream>

#include "progress_bar.hpp"
#include "imwrite_video.hpp"

#include <openvino/genai/video_generation/text2video_pipeline.hpp>
#include <openvino/genai/taylorseer_config.hpp>

int main(int argc, char* argv[]) try {
    OPENVINO_ASSERT(argc == 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>'");

    const std::string models_path = argv[1];
    const std::string prompt = argv[2];
    const std::string device = "CPU";  // GPU can be used as well
    const std::string negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted";
    const size_t num_inference_steps = 25;

    ov::genai::Text2VideoPipeline pipe(models_path, device);
    const size_t frame_rate = pipe.get_generation_config().frame_rate.value();
    std::cout << "Generating baseline video without caching...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    auto baseline_output = pipe.generate(
        prompt,
        ov::genai::negative_prompt(negative_prompt),
        ov::genai::num_inference_steps(num_inference_steps),
        ov::genai::callback(progress_bar)
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    auto baseline_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Baseline generation completed in " << baseline_duration.count() / 1000.0 << "s\n";

    save_video("taylorseer_baseline.avi", baseline_output.video, frame_rate);
    std::cout << "Baseline video saved to taylorseer_baseline.avi\n";

    // Configure TaylorSeer caching
    std::cout << "\nGenerating video with TaylorSeer caching...\n";
    const size_t cache_interval = 3;
    const size_t disable_before = 6;
    const int disable_after = -2;
    ov::genai::TaylorSeerCacheConfig taylorseer_config{cache_interval, disable_before, disable_after};
    std::cout << taylorseer_config.to_string() << "\n";
    auto generation_config = pipe.get_generation_config();
    generation_config.taylorseer_config = taylorseer_config;
    pipe.set_generation_config(generation_config);

    start_time = std::chrono::high_resolution_clock::now();

    auto output = pipe.generate(
        prompt,
        ov::genai::negative_prompt(negative_prompt),
        ov::genai::num_inference_steps(num_inference_steps),
        ov::genai::callback(progress_bar)
    );

    end_time = std::chrono::high_resolution_clock::now();
    auto taylorseer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "TaylorSeer generation completed in " << taylorseer_duration.count() / 1000.0 << "s\n";

    save_video("taylorseer.avi", output.video, frame_rate);
    std::cout << "Video saved to taylorseer.avi\n";

    // Performance comparison
    double baseline_ms = static_cast<double>(baseline_duration.count());
    double taylorseer_ms = static_cast<double>(taylorseer_duration.count());

    double speedup = taylorseer_ms > 0 ? baseline_ms / taylorseer_ms : 0.0;
    double time_saved = baseline_ms > 0 ? (baseline_ms - taylorseer_ms) / 1000.0 : 0.0;
    double percentage = baseline_ms > 0 ? (baseline_ms - taylorseer_ms) / baseline_ms * 100.0 : 0.0;

    std::cout << "\nPerformance Comparison:\n";
    std::cout << "  Baseline time: " << baseline_ms / 1000.0 << "s\n";
    std::cout << "  TaylorSeer time: " << taylorseer_ms / 1000.0 << "s\n";
    std::cout << "  Speedup: " << speedup << "x\n";
    std::cout << "  Time saved: " << time_saved << "s (" << percentage << "%)\n";

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
