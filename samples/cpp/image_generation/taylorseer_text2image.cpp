// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/text2image_pipeline.hpp"
#include "openvino/genai/taylorseer_config.hpp"

#include "imwrite.hpp"
#include "progress_bar.hpp"

#include <chrono>
#include <iostream>

int32_t main(int32_t argc, char* argv[]) try {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <MODEL_DIR> '<PROMPT>' [--cache-interval <N>] [--disable-before <N>] [--disable-after <N>] [--steps <N>]\n";
        std::cout << "\nTaylorSeer Cache Configurations:\n";
        std::cout << "  --cache-interval <N>   : Cache interval (default: 3)\n";
        std::cout << "  --disable-before <N>   : Disable caching before this step for warmup (default: 6)\n";
        std::cout << "  --disable-after <N>    : Disable caching after this step from end, -1 means last step (default: -2)\n";
        std::cout << "\nGeneration Options:\n";
        std::cout << "  --steps <N>            : Number of inference steps (default: 28)\n";
        return EXIT_FAILURE;
    }

    const std::string models_path = argv[1];
    const std::string prompt = argv[2];
    const std::string device = "CPU";  // GPU can be used as well

    // Parse optional arguments
    size_t cache_interval = 3;
    size_t disable_before = 6;
    int disable_after = -2;
    size_t num_inference_steps = 28;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cache-interval" && i + 1 < argc) {
            cache_interval = std::stoul(argv[++i]);
        } else if (arg == "--disable-before" && i + 1 < argc) {
            disable_before = std::stoul(argv[++i]);
        } else if (arg == "--disable-after" && i + 1 < argc) {
            disable_after = std::stoi(argv[++i]);
        } else if (arg == "--steps" && i + 1 < argc) {
            num_inference_steps = std::stoul(argv[++i]);
        }
    }

    ov::genai::Text2ImagePipeline pipe(models_path, device);
    std::cout << "Generating baseline image without caching...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    ov::Tensor baseline_image = pipe.generate(prompt,
        ov::genai::width(512),
        ov::genai::height(512),
        ov::genai::num_inference_steps(num_inference_steps),
        ov::genai::num_images_per_prompt(1),
        ov::genai::rng_seed(42),
        ov::genai::callback(progress_bar));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto baseline_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Baseline generation completed in " << baseline_duration.count() / 1000.0 << "s\n";

    imwrite("taylorseer_baseline.bmp", baseline_image, true);
    std::cout << "Baseline image saved to taylorseer_baseline.bmp\n";

    // Configure TaylorSeer caching
    std::cout << "\nGenerating image with TaylorSeer caching...\n";

    ov::genai::TaylorSeerCacheConfig taylorseer_config(cache_interval, disable_before, disable_after);
    std::cout << taylorseer_config.to_string() << "\n";
    auto generation_config = pipe.get_generation_config();
    generation_config.taylorseer_config = taylorseer_config;
    pipe.set_generation_config(generation_config);

    start_time = std::chrono::high_resolution_clock::now();

    ov::Tensor image = pipe.generate(prompt,
        ov::genai::width(512),
        ov::genai::height(512),
        ov::genai::num_inference_steps(num_inference_steps),
        ov::genai::num_images_per_prompt(1),
        ov::genai::rng_seed(42),
        ov::genai::callback(progress_bar));

    end_time = std::chrono::high_resolution_clock::now();
    auto taylorseer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "TaylorSeer generation completed in " << taylorseer_duration.count() / 1000.0 << "s\n";

    imwrite("taylorseer.bmp", image, true);
    std::cout << "Image saved to taylorseer.bmp\n";

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
