// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "audio_utils.hpp"
#include "load_image.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/omni/pipeline.hpp"
#include "openvino/genai/omni/talker_speech_config.hpp"

ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}

int main(int argc, char* argv[]) try {
    if (argc != 4) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE_OR_DIR> <AUDIO_FILE>");
    }

    const std::filesystem::path models_path = argv[1];

    // Two configs: text_config drives the thinker text decode, talker_speech_config drives
    // the talker + speech output. Speech output is hardcoded on here to show the multimodal
    // path. Set talker_speech_config.return_audio = false to get text-only responses.
    ov::genai::GenerationConfig text_config;
    text_config.max_new_tokens = 256;

    ov::genai::OmniTalkerSpeechConfig talker_speech_config(models_path);
    talker_speech_config.return_audio = true;
    // Leaving talker_speech_config.speaker empty selects the model's default voice. Available
    // voices vary by checkpoint.

    std::vector<ov::Tensor> rgbs = utils::load_images(argv[2]);

    ov::Tensor audio_tensor = utils::audio::read_wav_as_tensor(argv[3]);
    std::vector<ov::Tensor> audios = {std::move(audio_tensor)};

    // Compose OmniPipeline from separate VLM and Talker components.
    // This allows independent device selection (e.g., VLM on GPU, Talker on CPU),
    // sharing a VLM base across pipelines, or injecting custom TalkerBase implementations.
    // For simpler use cases: ov::genai::OmniPipeline pipe(models_path, "CPU");

    auto vlm = std::make_shared<ov::genai::VLMPipeline>(models_path, "CPU");
    auto talker = std::make_shared<ov::genai::Talker>(models_path, "CPU");
    ov::genai::OmniPipeline pipe(vlm, talker);

    std::cout << "OmniPipeline composed successfully.\n";

    // Speaker API demo: list available speakers and demonstrate voice blending.
    std::vector<std::string> speakers = talker->list_speakers();
    std::cout << "\n=== Available Speakers ===\n";
    if (speakers.empty()) {
        std::cout << "No named speakers (model has single default voice)\n";
    } else {
        std::cout << "Found " << speakers.size() << " speakers:\n";
        for (const auto& name : speakers) {
            std::cout << "  - " << name << "\n";
        }
        std::cout << "\n";

        if (speakers.size() >= 2) {
            std::cout << "=== Voice Blending Demo ===\n";
            std::cout << "Blending " << speakers[0] << " + " << speakers[1] << " (50/50 mix)\n";
            ov::Tensor emb1 = talker->get_speaker_embedding(speakers[0]);
            ov::Tensor emb2 = talker->get_speaker_embedding(speakers[1]);
            ov::Tensor blended(emb1.get_element_type(), emb1.get_shape());
            const float* data1 = emb1.data<float>();
            const float* data2 = emb2.data<float>();
            float* blended_data = blended.data<float>();
            for (size_t i = 0; i < emb1.get_size(); ++i) {
                blended_data[i] = 0.5f * data1[i] + 0.5f * data2[i];
            }
            talker_speech_config.speaker = blended;
            std::cout << "Using blended voice for generation.\n\n";
        }
    }

    ov::genai::ChatHistory history;
    std::vector<ov::Tensor> videos;  // Empty: sample uses images and audio only
    std::vector<ov::genai::VideoMetadata> videos_metadata;

    std::string prompt;
    std::cout << "question:\n";
    std::getline(std::cin, prompt);

    history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
    ov::genai::OmniDecodedResults decoded_results = pipe.generate(history,
                                                                   rgbs,
                                                                   videos,
                                                                   videos_metadata,
                                                                   audios,
                                                                   text_config,
                                                                   talker_speech_config,
                                                                   print_subword);
    history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});

    if (!decoded_results.speech_result.waveforms.empty()) {
        std::cout << "\n[Speech output: " << decoded_results.speech_result.waveforms[0].get_size() << " samples at 24kHz]"
                  << std::endl;
    }

    std::cout << "\n----------\n"
                 "question:\n";
    while (std::getline(std::cin, prompt)) {
        history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
        // New images and audio can be passed at each turn; here we rely on the info from turn 1.
        std::vector<ov::Tensor> images, turn_audios;
        decoded_results = pipe.generate(history,
                                        images,
                                        videos,
                                        videos_metadata,
                                        turn_audios,
                                        text_config,
                                        talker_speech_config,
                                        print_subword);
        history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});

        if (!decoded_results.speech_result.waveforms.empty()) {
            std::cout << "\n[Speech output: " << decoded_results.speech_result.waveforms[0].get_size() << " samples at 24kHz]"
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
