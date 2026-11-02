// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/omni/pipeline.hpp"
#include "openvino/genai/omni/talker_speech_config.hpp"
#include "openvino/genai/visual_language/video_metadata.hpp"
#include "speech_generation/audio_utils.hpp"
#include "visual_language_chat/load_image.hpp"
#include "visual_language_chat/load_video.hpp"

namespace {
constexpr uint32_t SPEECH_SAMPLE_RATE = 24000;  // Qwen3-Omni and LFM3-Omni speech output is 24kHz mono PCM.

ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}

// Save the first waveform of a result to a WAV file. Speech output is optional,
// so callers pass results whose waveforms may be empty.
void save_speech(const ov::genai::OmniDecodedResults& results, const std::string& file_name) {
    if (results.speech_result.waveforms.empty()) {
        return;
    }
    const ov::Tensor& waveform = results.speech_result.waveforms[0];
    utils::audio::save_to_wav(waveform.data<const float>(),
                              waveform.get_size(),
                              file_name,
                              waveform.get_element_type().bitwidth(),
                              SPEECH_SAMPLE_RATE);
}
}  // namespace

int main(int argc, char* argv[]) try {
    if (argc != 5) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                 " <MODEL_DIR> <IMAGE_FILE_OR_DIR> <AUDIO_FILE> <VIDEO_FILE>");
    }

    const std::filesystem::path models_path = argv[1];

    // Two configs: text_config drives the text decode, talker_speech_config drives the speech
    // output. Speech output is hardcoded on here to show the multimodal
    // path. Set talker_speech_config.return_audio = false to get text-only responses.
    ov::genai::GenerationConfig text_config;
    text_config.max_new_tokens = 256;

    ov::genai::OmniTalkerSpeechConfig talker_speech_config(models_path);
    talker_speech_config.return_audio = true;
    // Leaving talker_speech_config.speaker empty selects the model's default voice. Qwen3-Omni
    // voices vary by checkpoint; LFM3-Omni takes the voice from the system prompt.

    std::vector<ov::Tensor> rgbs = utils::load_images(argv[2]);

    ov::Tensor audio_tensor = utils::audio::read_wav_as_tensor(argv[3]);
    std::vector<ov::Tensor> audios = {std::move(audio_tensor)};

    ov::genai::OmniPipeline pipe(models_path, "CPU");

    ov::genai::ChatHistory history;
    auto [video, video_metadata] = utils::load_video(argv[4]);
    std::vector<ov::Tensor> videos;
    videos.push_back(std::move(video));
    std::vector<ov::genai::VideoMetadata> videos_metadata;
    videos_metadata.push_back(std::move(video_metadata));

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
    save_speech(decoded_results, "output_audio_0.wav");

    std::cout << "\n----------\n"
                 "question:\n";
    for (size_t turn = 1; std::getline(std::cin, prompt); ++turn) {
        history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
        // Media attaches to the turn it is supplied on, so later turns pass none at all and refer
        // back through the history rather than re-sending turn 1's tensors.
        std::vector<ov::Tensor> turn_images, turn_videos, turn_audios;
        std::vector<ov::genai::VideoMetadata> turn_videos_metadata;
        decoded_results = pipe.generate(history,
                                        turn_images,
                                        turn_videos,
                                        turn_videos_metadata,
                                        turn_audios,
                                        text_config,
                                        talker_speech_config,
                                        print_subword);
        history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});
        save_speech(decoded_results, "output_audio_" + std::to_string(turn) + ".wav");

        std::cout << "\n----------\n"
                     "question:\n";
    }
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
