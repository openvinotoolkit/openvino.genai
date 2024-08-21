// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/audio_utils.hpp"
#include "openvino/genai/whisper_speech_recognition_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (3 > argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<WAV_FILE_PATH>\"");
    }

    std::vector<float> pcmf32;                // mono-channel F32 PCM
    std::vector<std::vector<float>> pcmf32s;  // stereo-channel F32 PCM

    if (!ov::genai::utils::audio::read_wav(std::string{argv[2]}, pcmf32, pcmf32s, false)) {
        throw std::runtime_error("Failed to read WAV file " + std::string{argv[2]});
    }

    auto streamer = [](std::string subword) {
        std::cout << subword << std::flush;
        return false;
    };

    ov::genai::WhisperSpeechRecognitionPipeline pipeline{std::string{argv[1]}};

    ov::genai::WhisperGenerationConfig config = pipeline.get_generation_config();
    // config.max_new_tokens = 15;

    auto results = pipeline.generate(pcmf32, config, streamer);

    std::cout << results << std::endl;

    // std::string model_path = argv[1];
    // std::string prompt = argv[2];
    // std::string device = "CPU";  // GPU can be used as well

    // ov::genai::LLMPipeline pipe(model_path, device);
    // ov::genai::GenerationConfig config;
    // config.max_new_tokens = 100;
    // std::string result = pipe.generate(prompt, config);
    // std::cout << result << std::endl;
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
