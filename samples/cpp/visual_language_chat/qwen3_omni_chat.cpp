// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "load_image.hpp"
#include "openvino/genai/omni/pipeline.hpp"
#include "openvino/genai/omni/speech_generation_config.hpp"

ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}

int main(int argc, char* argv[]) try {
    if (argc != 3) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE_OR_DIR>");
    }

    const std::filesystem::path models_path = argv[1];

    // Speech output is hardcoded here to show the multimodal path.
    // Set return_audio = false to get text-only responses.
    ov::genai::OmniSpeechGenerationConfig speech_config(models_path);
    speech_config.max_new_tokens = 256;
    speech_config.return_audio = true;
    // Leaving speaker empty selects the model's default voice. Available voices vary by checkpoint
    // (e.g. MoE exposes "Ethan", "Chelsie", "Aiden", "Cherry"); the full list is in
    // talker_config.speaker_id of the model's config.json.

    std::vector<ov::Tensor> rgbs = utils::load_images(argv[2]);

    ov::genai::OmniPipeline pipe(models_path, "CPU");

    ov::genai::ChatHistory history;

    std::string prompt;
    std::cout << "question:\n";
    std::getline(std::cin, prompt);

    history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
    ov::genai::OmniDecodedResults decoded_results = pipe.generate(history,
                                                                   rgbs,
                                                                   /*videos=*/{},
                                                                   /*audios=*/{},
                                                                   speech_config,
                                                                   print_subword);
    history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});

    if (!decoded_results.speech_outputs.empty()) {
        std::cout << "\n[Speech output: " << decoded_results.speech_outputs[0].get_size() << " samples at 24kHz]"
                  << std::endl;
    }

    std::cout << "\n----------\n"
                 "question:\n";
    while (std::getline(std::cin, prompt)) {
        history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
        // New images can be passed at each turn; here we reuse the initial one only on turn 1.
        decoded_results = pipe.generate(history,
                                        /*images=*/{},
                                        /*videos=*/{},
                                        /*audios=*/{},
                                        speech_config,
                                        print_subword);
        history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});

        if (!decoded_results.speech_outputs.empty()) {
            std::cout << "\n[Speech output: " << decoded_results.speech_outputs[0].get_size() << " samples at 24kHz]"
                      << std::endl;
        }
        std::cout << "\n----------\n"
                     "question:\n";
    }
    return 0;
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
