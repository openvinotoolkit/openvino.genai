// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <iostream>
#include <openvino/genai/visual_language/pipeline.hpp>

#include "audio_utils.hpp"

ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}

int main(int argc, char* argv[]) try {
    if (argc < 3 || argc > 4) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <AUDIO_FILE> [DEVICE]");
    }

    ov::Tensor audio = utils::audio::read_wav_as_tensor(argv[2]);

    // GPU and NPU can be used as well.
    // Note: If NPU is selected, only language model will be run on NPU
    std::string device = (argc >= 4) ? argv[3] : "CPU";

    ov::AnyMap properties;
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        properties.insert({ov::cache_dir("vlm_cache")});
    }
    ov::genai::VLMPipeline pipe(argv[1], device, properties);

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;

    std::string prompt;

    ov::genai::ChatHistory history;

    std::cout << "question:\n";
    std::getline(std::cin, prompt);

    history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
    ov::genai::VLMDecodedResults decoded_results = pipe.generate(history,
                                                                 ov::genai::audios(std::vector{audio}),
                                                                 ov::genai::generation_config(generation_config),
                                                                 ov::genai::streamer(print_subword));
    history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});
    std::cout << "\n----------\n"
                 "question:\n";
    while (std::getline(std::cin, prompt)) {
        history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
        // New images and videos can be passed at each turn
        ov::genai::VLMDecodedResults decoded_results =
            pipe.generate(history, ov::genai::generation_config(generation_config), ov::genai::streamer(print_subword));
        history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});
        std::cout << "\n----------\n"
                     "question:\n";
    }
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
