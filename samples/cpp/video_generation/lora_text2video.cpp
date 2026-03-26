// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <openvino/genai/video_generation/text2video_pipeline.hpp>
#include <string>

#include "imwrite_video.hpp"
#include "progress_bar.hpp"

void print_perf_metrics(ov::genai::VideoGenerationPerfMetrics& perf_metrics) {
    std::cout << "\nPerformance metrics:\n"
              << "  Load time: " << perf_metrics.get_load_time() << " ms\n"
              << "  Generate duration: " << perf_metrics.get_generate_duration() << " ms\n"
              << "  Transformer duration: " << perf_metrics.get_transformer_infer_duration().mean << " ms\n"
              << "  VAE decoder duration: " << perf_metrics.get_vae_decoder_infer_duration() << " ms\n";
}

int main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 3 && (argc - 3) % 2 == 0,
                    "Usage: ",
                    argv[0],
                    " <MODEL_DIR> '<PROMPT>' [<LORA_SAFETENSORS> <ALPHA> ...]");

    std::filesystem::path models_dir = argv[1];
    std::string prompt = argv[2];

    const std::string device = "CPU";  // GPU can be used as well

    ov::genai::AdapterConfig adapter_config;
    // Multiple LoRA adapters applied simultaneously are supported, parse them all and corresponding alphas from cmd
    // parameters:
    for (size_t i = 0; i < (argc - 3) / 2; ++i) {
        ov::genai::Adapter adapter(argv[3 + 2 * i]);
        float alpha = std::atof(argv[3 + 2 * i + 1]);
        adapter_config.add(adapter, alpha);
    }

    // LoRA adapters passed to the constructor will be activated by default in next generates
    ov::genai::Text2VideoPipeline pipe(models_dir, device, ov::genai::adapters(adapter_config));

    const float frame_rate = 25.0f;

    std::cout << "Generating video with LoRA adapters applied, resulting video will be in lora_video.avi\n";
    auto output =
        pipe.generate(prompt,
                      ov::genai::negative_prompt("worst quality, inconsistent motion, blurry, jittery, distorted"),
                      ov::genai::height(480),
                      ov::genai::num_inference_steps(25),
                      ov::genai::callback(progress_bar),
                      ov::genai::guidance_scale(3));

    save_video("lora_video.avi", output.video, frame_rate);
    print_perf_metrics(output.performance_stat);

    std::cout << "Generating video without LoRA adapters applied, resulting video will be in baseline_video.avi\n";
    output = pipe.generate(prompt,
                           ov::genai::adapters(),  // passing adapters in generate overrides adapters set in the
                                                   // constructor; adapters() means no adapters
                           ov::genai::negative_prompt("worst quality, inconsistent motion, blurry, jittery, distorted"),
                           ov::genai::height(480),
                           ov::genai::num_inference_steps(25),
                           ov::genai::callback(progress_bar),
                           ov::genai::guidance_scale(3));

    save_video("baseline_video.avi", output.video, frame_rate);
    print_perf_metrics(output.performance_stat);

    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}
