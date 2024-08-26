// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"
#include "openvino/genai/whisper_speech_recognition_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (3 > argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<WAV_FILE_PATH>\"");
    }

    std::string model_path = argv[1];
    std::string wav_file_path = argv[2];

    std::vector<float> raw_speech;                 // mono-channel F32 PCM
    std::vector<std::vector<float>> raw_speech_s;  // stereo-channel F32 PCM

    if (!utils::audio::read_wav(wav_file_path, raw_speech, raw_speech_s)) {
        throw std::runtime_error("Failed to read WAV file " + wav_file_path);
    }

    ov::genai::WhisperSpeechRecognitionPipeline pipeline{model_path};

    auto results = pipeline.generate(raw_speech);

    std::cout << results << std::endl;

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
