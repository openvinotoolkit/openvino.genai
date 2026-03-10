// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}
int main(int argc, char* argv[]) try {
    // Usage: app <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES> <DEVICE> [<LORA_SAFETENSORS> <ALPHA> ...]
    if (argc < 4 || ((argc - 4) % 2) != 0) {
        throw std::runtime_error(
            std::string{"Usage "} + argv[0] +
            " <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES> <DEVICE> [<LORA_SAFETENSORS> <ALPHA> ...]"
        );
    }

    std::vector<ov::Tensor> rgbs = utils::load_images(argv[2]);

    std::string device = argv[3];
    ov::AnyMap pipeline_properties;
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        pipeline_properties.insert({ov::cache_dir("vlm_cache")});
    }

    // LoRA args parsed as pairs: <LORA_SAFETENSORS> <ALPHA>
    ov::genai::AdapterConfig adapter_config;
    if (argc > 4) {
        for (int idx = 4; idx + 1 < argc; idx += 2) {
            ov::genai::Adapter adapter(argv[idx]);
            float alpha = std::stof(argv[idx + 1]);
            adapter_config.add(adapter, alpha);
        }
        pipeline_properties.insert({ov::genai::adapters(adapter_config)});
    }

    ov::genai::VLMPipeline pipe(argv[1], device, pipeline_properties);

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;

    std::string prompt;

    std::cout << "question:\n";
    std::getline(std::cin, prompt);

    std::cout << "----------\nGenerating answer with LoRA adapters applied:\n";
    pipe.generate(prompt,
                  ov::genai::images(rgbs),
                  ov::genai::generation_config(generation_config),
                  ov::genai::streamer(print_subword));

    std::cout << "\n----------\nGenerating answer without LoRA adapters applied:\n";
    pipe.generate(prompt,
                  ov::genai::images(rgbs),
                  ov::genai::generation_config(generation_config),
                  ov::genai::adapters(),
                  ov::genai::streamer(print_subword));
    std::cout << "\n----------\n";

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
