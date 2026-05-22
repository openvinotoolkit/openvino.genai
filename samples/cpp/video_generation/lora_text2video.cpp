// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <filesystem>

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
    OPENVINO_ASSERT(opts.positional.size() >= 2 && (opts.positional.size() - 2) % 2 == 0,
                    "Usage: ", argv[0],
                    " <MODEL_DIR> '<PROMPT>' [<LORA_SAFETENSORS> <ALPHA> ...]"
                    " [--height H] [--width W] [--num-frames N] [--num-inference-steps S]");

    std::filesystem::path models_dir = opts.positional[0];
    std::string prompt = opts.positional[1];

    const std::string device = "CPU";  // GPU can be used as well

    ov::genai::AdapterConfig adapter_config;
    // Multiple LoRA adapters applied simultaneously are supported, parse them all and corresponding alphas from cmd parameters:
    for (size_t i = 0; i < (opts.positional.size() - 2) / 2; ++i) {
        ov::genai::Adapter adapter(opts.positional[2 + 2*i]);
        float alpha = std::atof(opts.positional[2 + 2*i + 1].c_str());
        adapter_config.add(adapter, alpha);
    }

    // LoRA adapters passed to the constructor will be activated by default in next generates
    ov::genai::Text2VideoPipeline pipe(models_dir, device, ov::genai::adapters(adapter_config));

    const float frame_rate = 25.0f;

    const int64_t height_v = opts.height.value_or(480);
    const size_t num_inference_steps_v = static_cast<size_t>(opts.num_inference_steps.value_or(25));

    auto generate = [&](const ov::genai::AdapterConfig* override_adapters) {
        ov::AnyMap args{
            {"negative_prompt", std::string("worst quality, inconsistent motion, blurry, jittery, distorted")},
            {"height", height_v},
            {"num_inference_steps", num_inference_steps_v},
            {"callback", std::function<bool(size_t, size_t, ov::Tensor&)>(progress_bar)},
            {"guidance_scale", 3.0f},
        };
        if (opts.width.has_value()) args["width"] = static_cast<int64_t>(*opts.width);
        if (opts.num_frames.has_value()) args["num_frames"] = static_cast<size_t>(*opts.num_frames);
        if (override_adapters != nullptr) args["adapters"] = *override_adapters;
        return pipe.generate(prompt, args);
    };

    std::cout << "Generating video with LoRA adapters applied, resulting video will be in lora_video.avi\n";
    auto output = generate(nullptr);

    save_video("lora_video.avi", output.video, frame_rate);
    print_perf_metrics(output.performance_stat);

    std::cout << "Generating video without LoRA adapters applied, resulting video will be in baseline_video.avi\n";
    // passing adapters in generate overrides adapters set in the constructor; an empty AdapterConfig means no adapters
    ov::genai::AdapterConfig no_adapters;
    output = generate(&no_adapters);

    save_video("baseline_video.avi", output.video, frame_rate);
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
