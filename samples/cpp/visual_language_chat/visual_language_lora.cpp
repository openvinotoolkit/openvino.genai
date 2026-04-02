// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <openvino/core/except.hpp>
#include <openvino/genai/visual_language/pipeline.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "load_image.hpp"

ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}
int main(int argc, char* argv[]) try {
    // At least one LoRA adapter must be provided.
    OPENVINO_ASSERT(
        argc >= 6 && ((argc - 4) % 2) == 0,
        "Usage: ",
        argv[0],
        " <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES> <PROMPT> <LORA_SAFETENSORS> <ALPHA> [<LORA_SAFETENSORS> <ALPHA> "
        "...]"
    );

    std::vector<ov::Tensor> rgbs = utils::load_images(argv[2]);

    const std::string device = "CPU";  // GPU can be used as well
    ov::AnyMap pipeline_properties;

    const std::string prompt = argv[3];

    // LoRA args parsed as pairs: <LORA_SAFETENSORS> <ALPHA>
    ov::genai::AdapterConfig adapter_config;
    for (int idx = 4; idx + 1 < argc; idx += 2) {
        ov::genai::Adapter adapter(argv[idx]);
        float alpha = std::stof(argv[idx + 1]);
        adapter_config.add(adapter, alpha);
    }
    pipeline_properties.insert({ov::genai::adapters(adapter_config)});

    ov::genai::VLMPipeline pipe(argv[1], device, pipeline_properties);

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;

    std::cout << "Generating answer with LoRA adapters applied:\n";
    pipe.generate(
        prompt,
        ov::genai::images(rgbs),
        ov::genai::generation_config(generation_config),
        ov::genai::streamer(print_subword)
    );

    std::cout << "\n----------\nGenerating answer without LoRA adapters applied:\n";
    pipe.generate(
        prompt,
        ov::genai::images(rgbs),
        ov::genai::generation_config(generation_config),
        ov::genai::adapters(),
        ov::genai::streamer(print_subword)
    );
    std::cout << "\n----------\n";

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
