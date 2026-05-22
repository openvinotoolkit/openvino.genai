// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>
#include <random>
#include <filesystem>
#include <iostream>

#include "progress_bar.hpp"
#include "imwrite_video.hpp"
#include "video_args.hpp"

#include <openvino/genai/video_generation/text2video_pipeline.hpp>

void print_perf_metrics(ov::genai::VideoGenerationPerfMetrics& perf_metrics) {
    std::cout << "\nPerformance metrics:\n"
              << "  Load time: " << perf_metrics.get_load_time() << " ms\n"
              << "  Generate duration: " << perf_metrics.get_generate_duration() << " ms\n"
              << "  Transformer duration: " << perf_metrics.get_transformer_infer_duration().mean << " ms\n"
              << "  VAE decoder duration: " << perf_metrics.get_vae_decoder_infer_duration() << " ms\n";
}

int main(int32_t argc, char* argv[]) try {
    auto opts = video_args::parse(argc, argv);
    OPENVINO_ASSERT(opts.positional.size() == 2,
                    "Usage: ", argv[0],
                    " <MODEL_DIR> '<PROMPT>'"
                    " [--height H] [--width W] [--num-frames N] [--num-inference-steps S]");

    std::filesystem::path models_dir = opts.positional[0];
    std::string prompt = opts.positional[1];

    const std::string device = "CPU";  // GPU can be used as well
    float frame_rate = 25.0f;

    ov::genai::Text2VideoPipeline pipe(models_dir, device);
    auto output = pipe.generate(
        prompt,
        ov::genai::negative_prompt("worst quality, inconsistent motion, blurry, jittery, distorted"),
        ov::genai::height(opts.height.value_or(480)),
        ov::genai::width(opts.width.value_or(704)),
        ov::genai::num_frames(opts.num_frames.value_or(161)),
        ov::genai::num_inference_steps(opts.num_inference_steps.value_or(25)),
        ov::genai::num_videos_per_prompt(1),
        ov::genai::callback(progress_bar),
        ov::genai::frame_rate(frame_rate),
        ov::genai::guidance_scale(3)
    );

    save_video("genai_video.avi", output.video, frame_rate);
    print_perf_metrics(output.performance_stat);

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
